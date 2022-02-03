import os
import pickle
import zipfile
from typing import List, Dict

import click
import fire
import tqdm

import utils
from databutler.datana.generic.corpus.code import DatanaFunction
from databutler.datana.generic.corpus.processing.base_processor import DatanaFunctionProcessorChain
from databutler.datana.viz.corpus.code_processors import VizMplKeywordArgNormalizer, VizMplFuncNameExtractor, \
    VizMplVarNameOptimizer, VizMplAxesCounter, VizMplCodeNormalizer
from databutler.utils import paths, gdriveutils, pickleutils, libversioning, multiprocess
from databutler.utils.logging import logger


class CommandNotFinishedError(Exception):
    pass


def _get_vizsmith_data_path() -> str:
    path = os.path.join(paths.get_user_home_dir_path(), ".databutler", "data", "vizsmith")
    os.makedirs(path, exist_ok=True)
    return path


class _RawVizSmithData:
    DOWNLOAD_CMD = "download_raw_data"

    FOLDER_NAME = "RawData"
    GDRIVE_ID = "1hUWxzT15SFKP9WS8F-QxVntc-zRF9WuX"
    DB_NAME = "vizsmith_raw_db.pkl"
    DB_DF_ARGS_NAME = "vizsmith_raw_db_df_args.zip"

    @staticmethod
    def get_raw_data_path() -> str:
        return os.path.join(_get_vizsmith_data_path(), _RawVizSmithData.FOLDER_NAME)

    @staticmethod
    @utils.fire_command(name=DOWNLOAD_CMD, collection=__file__)
    def download():
        """
        Downloads raw VizSmith data from https://drive.google.com/drive/u/1/folders/1hUWxzT15SFKP9WS8F-QxVntc-zRF9WuX.
        """
        path = _get_vizsmith_data_path()
        raw_data_path = _RawVizSmithData.get_raw_data_path()

        download = True
        if utils.check_drive_contents_against_path(_RawVizSmithData.GDRIVE_ID, raw_data_path):
            logger.info(f"Raw VizSmith data already exists at {raw_data_path}")
            if not click.confirm("Do you want to re-download the raw data for VizSmith?"):
                download = False

        if download:
            gdriveutils.download_folder(_RawVizSmithData.GDRIVE_ID, path)

        assert os.path.exists(raw_data_path)

    @staticmethod
    def check_finished():
        raw_data_path = _RawVizSmithData.get_raw_data_path()

        if not utils.check_drive_contents_against_path(_RawVizSmithData.GDRIVE_ID, raw_data_path):
            raise CommandNotFinishedError(f"Raw data not upto date. Please run `{_RawVizSmithData.DOWNLOAD_CMD}`")


class _VizSmithDataPrep:
    PREPARE_CMD = "prepare_raw_data"

    FOLDER_NAME = "PreparedData"
    FNS_FILE_NAME = "vizsmith_datana_funcs.pkl"
    ARGS_FILE_NAME = "vizsmith_db_args.pkl"

    @staticmethod
    def get_prepared_data_path() -> str:
        path = os.path.join(_get_vizsmith_data_path(), _VizSmithDataPrep.FOLDER_NAME)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_datana_funcs_path() -> str:
        return os.path.join(_VizSmithDataPrep.get_prepared_data_path(), _VizSmithDataPrep.FNS_FILE_NAME)

    @staticmethod
    def get_args_path() -> str:
        return os.path.join(_VizSmithDataPrep.get_prepared_data_path(), _VizSmithDataPrep.ARGS_FILE_NAME)

    @staticmethod
    @utils.fire_command(name=PREPARE_CMD, collection=__file__)
    def prepare_raw_vizsmith_data():
        """
        Converts raw VizSmith data into the databutler internal representation.
        """
        _RawVizSmithData.check_finished()

        raw_data_path = _RawVizSmithData.get_raw_data_path()

        db_path = os.path.join(raw_data_path, _RawVizSmithData.DB_NAME)
        args_path = _VizSmithDataPrep.get_args_path()
        fns_path = _VizSmithDataPrep.get_datana_funcs_path()

        if os.path.exists(args_path) and os.path.exists(fns_path):
            logger.info("Found existing preparation of raw VizSmith data")
            if not click.confirm("Do you want to prepare raw VizSmith data again?"):
                return

        raw_db = pickleutils.smart_load(db_path)

        #  Remove non-reusable functions
        raw_db = [f for f in raw_db if f['reusable']]
        logger.info(f"Found {len(raw_db)} VizSmith functions")

        #  Convert these to DatanaFunctions
        datana_functions: List[DatanaFunction] = []
        for fn in tqdm.tqdm(raw_db, desc="Converting to Datana Functions", dynamic_ncols=True):
            fn = fn.copy()
            datana_functions.append(DatanaFunction(
                code_str=fn.pop('code'),
                uid=fn.pop('uid'),
                #  We know that the function names are all "visualization"
                func_name="visualization",
                #  Empty for now, we will populate this shortly
                pos_args=[],
                #  Empty for now, we will populate this shortly
                kw_args={},
                metadata={
                    **fn
                },
            ))

        #  Simplify lookup for future use.
        uid_to_datana_func: Dict[str, DatanaFunction] = {d_fn.uid: d_fn for d_fn in datana_functions}

        #  We now populate the Datana functions with the arguments.

        #  First open the zip file. Also open the a pickled collection writer to store the args within databutler.
        df_args_zip_path = os.path.join(raw_data_path, _RawVizSmithData.DB_DF_ARGS_NAME)
        pickled_collection_path = _VizSmithDataPrep.get_args_path()
        uid_to_pickled_refs: Dict[str, pickleutils.PickledRef] = {}

        with zipfile.ZipFile(df_args_zip_path) as df_args_zip, \
                pickleutils.PickledCollectionWriter(pickled_collection_path, overwrite_existing=True) as writer:
            #  Find all the *.pkl files. These correspond to the dataframe args.
            pkl_files = [p for p in df_args_zip.namelist() if p.endswith(".pkl")]

            ref_index = 0
            for pkl_path in tqdm.tqdm(pkl_files, desc="Collecting Args", dynamic_ncols=True):
                #  This corresponds to the UID of the visualization function.
                uid = os.path.basename(pkl_path).split(".pkl")[0]
                if uid not in uid_to_datana_func:
                    logger.warning(f"No datana function was found with UID {uid}")
                    continue

                with df_args_zip.open(pkl_path, "r") as f:
                    df_args = pickle.load(f)

                #  Col args should already be available in the metadata
                col_args = uid_to_datana_func[uid].metadata["col_args"]

                args = {
                    **df_args,
                    **col_args,
                }

                #  Dump to the pickled collection.
                writer.append(args)

                #  Store a pickled ref
                uid_to_pickled_refs[uid] = pickleutils.PickledRef(pickled_collection_path, index=ref_index)
                ref_index += 1

        final_fns = []
        for uid, d_fn in uid_to_datana_func.items():
            if uid not in uid_to_pickled_refs:
                logger.warning(f"No args found for datana function with UID {uid}")
                continue

            d_fn.pos_args = []
            #  Store the keyword args as an ObjRef
            d_fn.kw_args = uid_to_pickled_refs[uid]

            final_fns.append(d_fn)

        logger.info("Finished preparing Datana functions")

        #  Store the completed datana functions in another pickle file.
        pickleutils.smart_dump(final_fns, fns_path)
        logger.info(f"Saved datana functions at {fns_path}")

    @staticmethod
    def check_finished():
        args_path = _VizSmithDataPrep.get_args_path()
        fns_path = _VizSmithDataPrep.get_datana_funcs_path()

        if not (os.path.exists(args_path) and os.path.exists(fns_path)):
            raise CommandNotFinishedError(f"Raw data not prepared.. Please run `{_VizSmithDataPrep.PREPARE_CMD}`")


class _CodeProcessing:
    PROCESSING_CMD = "process_datana_functions"

    FOLDER_NAME = "ProcessedData"
    FNS_FILE_NAME = "processed_vizsmith_datana_funcs.pkl"

    @staticmethod
    def get_processed_data_path() -> str:
        path = os.path.join(_get_vizsmith_data_path(), _CodeProcessing.FOLDER_NAME)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_processed_funcs_path() -> str:
        return os.path.join(_CodeProcessing.get_processed_data_path(), _CodeProcessing.FNS_FILE_NAME)

    @staticmethod
    def _setup_libversioning():
        #  Use this to make sure library versions are downloaded before multiprocessing.
        mpl_version = "3.2.1"
        seaborn_version = "0.10.0"

        if not libversioning._check_if_lib_version_exists("matplotlib", mpl_version):
            libversioning._download_lib_version("matplotlib", mpl_version)

        if not libversioning._check_if_lib_version_exists("seaborn", seaborn_version):
            libversioning._download_lib_version("seaborn", seaborn_version)

    @staticmethod
    def _processor_runner(func: DatanaFunction) -> DatanaFunction:
        #  These were the versions used in VizSmith
        mpl_version = "3.2.1"
        seaborn_version = "0.10.0"

        with libversioning.modified_lib_env("matplotlib", mpl_version), \
                libversioning.modified_lib_env("seaborn", seaborn_version):
            processor_chain = DatanaFunctionProcessorChain(
                processors=[
                    VizMplCodeNormalizer(),
                    VizMplAxesCounter(),
                    VizMplFuncNameExtractor(),
                    VizMplKeywordArgNormalizer(),
                    VizMplVarNameOptimizer(),
                ]
            )

            return processor_chain.run(func)

    @staticmethod
    @utils.fire_command(name=PROCESSING_CMD, collection=__file__)
    def process_datana_functions(num_processes: int = 2):
        """
        Runs the relevant code processors on the prepared datana visualization functions.

        Args:
            num_processes: Number of processes to use to speed up the workflow.
        """
        _VizSmithDataPrep.check_finished()

        prepared_fns_path = _VizSmithDataPrep.get_datana_funcs_path()
        processed_fns_path = _CodeProcessing.get_processed_funcs_path()

        if os.path.exists(processed_fns_path):
            logger.info("Found existing processing of prepared VizSmith data")
            if not click.confirm("Do you want to process prepared VizSmith data again?"):
                return

        prepared_fns = pickleutils.smart_load(prepared_fns_path)
        logger.info(f"Processing {len(prepared_fns)} prepared VizSmith/Datana functions ...")

        logger.info(f"Setting up libversioning ...")
        _CodeProcessing._setup_libversioning()

        processed_fn_results: List[multiprocess.TaskResult] = multiprocess.run_tasks_in_parallel(
            func=_CodeProcessing._processor_runner,
            tasks=prepared_fns,
            num_workers=num_processes,
            timeout_per_task=120,
            use_progress_bar=True,
            progress_bar_desc="Processing Prepared Functions",
            max_tasks_per_worker=1,  # Avoid environment sharing between the tasks
            use_spawn=True,
        )

        logger.info(f"Finished processing {len(processed_fn_results)} functions")
        total = len(processed_fn_results)
        successes = [tr.result for tr in processed_fn_results if tr.is_success()]
        num_timeouts = sum(1 for tr in processed_fn_results if tr.is_timeout())
        num_exceptions = sum(1 for tr in processed_fn_results if tr.is_exception())
        num_other = total - len(successes) - num_timeouts - num_exceptions

        logger.info(f"Successful: {len(successes)} | "
                    f"Timeouts: {num_timeouts} | "
                    f"Exceptions: {num_exceptions} | "
                    f"Other: {num_other}")

        pickleutils.smart_dump(successes, processed_fns_path)
        logger.info(f"Saved {len(successes)} processed functions to {processed_fns_path}")

    @staticmethod
    def check_finished():
        processed_fns_path = _CodeProcessing.get_processed_funcs_path()

        if not os.path.exists(processed_fns_path):
            raise CommandNotFinishedError(f"Prepared data not prcocessed. "
                                          f"Please run `{_CodeProcessing.PROCESSING_CMD}`")


if __name__ == "__main__":
    #  This was the version the VizSmith db was stored with.
    with libversioning.modified_lib_env("pandas", "1.1.3"):
        fire.Fire(utils.get_fire_commands_for_collection(__file__))
