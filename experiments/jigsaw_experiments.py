import ast
import json
import os
import random
import shutil
from typing import Dict, List, Any, Callable, Collection, Optional, Set, Union

import astunparse
import attrs
import fire
import numpy as np
import pandas as pd
import tqdm
from databutler.mining.static_pandas_mining.autodoc_result import AutodocResult
import torch
from simplet5 import SimpleT5
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration

from databutler.utils import langmodels, pickleutils
from databutler.utils.logging import logger
from databutler.utils.caching import cached_property
from databutler.mining.static_pandas_mining.autodoc_search import EmbeddingBasedSearcher


class Checker:
    @staticmethod
    def check(v1: Any, v2: Any) -> bool:
        return Checker.get_checker(v1)(v1, v2)

    @staticmethod
    def get_checker(v1: Any) -> Callable[[Any, Any], bool]:
        if isinstance(v1, pd.DataFrame):
            return Checker.check_dataframe
        if isinstance(v1, pd.Series):
            return Checker.check_series
        if isinstance(v1, pd.core.groupby.GroupBy):
            return Checker.check_groupby
        if isinstance(v1, np.ndarray):
            return Checker.check_ndarray
        if isinstance(v1, str):
            return Checker.check_default
        if isinstance(v1, Collection):
            return Checker.check_collection

        return Checker.check_default

    @staticmethod
    def check_dataframe(v1: pd.DataFrame, v2: Any) -> bool:
        if not isinstance(v2, pd.DataFrame):
            return False

        try:
            pd.testing.assert_frame_equal(
                v1, v2, check_names=False, check_dtype=False, check_like=True
            )
            return True
        except (AssertionError, TypeError, ValueError):
            return False
        except Exception as e:
            # logger.warn("DataFrame Comparison Failed")
            # logger.log(v1)
            # logger.log(v2)
            # logger.log(e)
            return False

    @staticmethod
    def check_series(v1: pd.Series, v2: Any) -> bool:
        if not isinstance(v2, pd.Series):
            return False

        try:
            pd.testing.assert_series_equal(v1, v2, check_dtype=False, check_names=False)
            return True
        except (AssertionError, TypeError, ValueError):
            return False
        except Exception as e:
            # logger.warn("Series Comparison Failed")
            # logger.log(v1)
            # logger.log(v2)
            # logger.log(e)
            return False

    @staticmethod
    def check_index(v1: pd.Index, v2: Any) -> bool:
        if not isinstance(v2, pd.Index):
            return False

        try:
            pd.testing.assert_index_equal(v1, v2)
            return True
        except (AssertionError, TypeError, ValueError):
            return False
        except Exception as e:
            # logger.warn("Index Comparison Failed")
            # logger.log(v1)
            # logger.log(v2)
            # logger.log(e)
            return False

    @staticmethod
    def check_ndarray(v1: np.ndarray, v2: Any) -> bool:
        if not isinstance(v2, np.ndarray):
            return False

        try:
            return np.array_equal(v1, v2)
        except Exception as e:
            # logger.warn("NDArray Comparison Failed")
            # logger.log(v1)
            # logger.log(v2)
            # logger.log(e)
            return False

    @staticmethod
    def check_collection(v1: Collection, v2: Any) -> bool:
        if (not isinstance(v2, Collection)) or isinstance(v2, str):
            return False

        if len(v1) != len(v2):
            return False

        try:
            for i, j in zip(v1, v2):
                if not Checker.check(i, j):
                    return False

        except Exception as e:
            # logger.warn("Collection Comparison Failed")
            # logger.log(v1)
            # logger.log(v2)
            # logger.log(e)
            return False

        return True

    @staticmethod
    def check_default(v1: Any, v2: Any) -> bool:
        try:
            if v1 == v2:
                #  I know what I'm doing
                #  v1 == v2 is not guaranteed to return a bool
                #  This is to capture that
                return True
            else:
                return False

        except (AssertionError, TypeError, ValueError, NameError):
            return False
        except Exception as e:
            # logger.warn("Default Comparison Failed")
            # logger.log(v1)
            # logger.log(v2)
            # logger.log(e)
            return False


def load_benchmarks_1() -> Dict:
    with open(os.path.join(os.path.dirname(__file__), "PandasEval1.json")) as f:
        return json.load(f)


def load_benchmarks_2() -> Dict:
    with open(os.path.join(os.path.dirname(__file__), "PandasEval2.json")) as f:
        return json.load(f)


@attrs.define(eq=False, repr=False, slots=False)
class CodexBaseline:
    results_save_path: str
    overwrite_existing_results: bool = False
    engine: str = "code-davinci-002"
    temperature: float = 0.0
    num_repetitions: int = 1
    ignore_eval_1: bool = False
    ignore_eval_2: bool = False
    run_id: Optional[str] = None

    def setup_context(self, io_ex: Dict):
        assert all(
            isinstance(ast.parse(expr).body[0], ast.Expr) for expr in io_ex["inputs"]
        )
        assert all(
            isinstance(ast.parse(expr).body[0], ast.Expr) for expr in [io_ex["output"]]
        )
        assert len(io_ex["inputs"]) == len(io_ex["invars"])
        context = "\n".join(
            f"{var} = {expr}" for var, expr in zip(io_ex["invars"], io_ex["inputs"])
        )
        return context

    def setup_prompt(self, query: str, ios: List[Dict]):
        context = self.setup_context(ios[0])
        prompt = f"{context}\n\n#  {query}\n"
        return prompt

    def get_expected_output(self, io_ex: Dict):
        state = {}
        imports = "\n".join(
            [
                "import pandas",
                "import pandas as pd",
                "import numpy",
                "import numpy as np",
            ]
        )
        exec(imports, state)
        return eval(io_ex["output"], state)

    def evaluate_result_on_io_ex(self, codex_resp: str, io_ex: Dict) -> bool:
        codex_resp = (
            codex_resp.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        )
        try:
            code_ast = ast.parse(codex_resp).body[0]
        except:
            return False

        if codex_resp.startswith("print("):
            assert isinstance(code_ast, ast.Expr)
            assert isinstance(code_ast.value, ast.Call)
            code_ast = code_ast.value.args[0]
            codex_resp = astunparse.unparse(code_ast)
            logger.debug(f"Unparsed: {codex_resp}")
            code_ast = ast.parse(codex_resp).body[0]

        code_context = self.setup_context(io_ex)
        state = {}
        imports = "\n".join(
            [
                "import pandas",
                "import pandas as pd",
                "import numpy",
                "import numpy as np",
            ]
        )
        exec(imports, state)
        exec(code_context, state)

        expected_output = self.get_expected_output(io_ex)
        res_cands = []

        try:
            if isinstance(code_ast, ast.Expr):
                res_cands = [eval(codex_resp, state)]
            else:
                exec(codex_resp, state)
        except:
            return False

        res_cands.extend(
            [
                v
                for v in state.values()
                if isinstance(
                    v,
                    (
                        type(expected_output),
                        int,
                        float,
                        str,
                    ),
                )
            ]
        )

        print(res_cands)
        print(expected_output)

        if any(Checker.check(res, expected_output) for res in res_cands):
            return True

    def evaluate_result(self, codex_resp: str, ios: List[Dict]) -> bool:
        return any(self.evaluate_result_on_io_ex(codex_resp, io) for io in ios)

    def run_benchmark(self, query: str, ios: List[Dict], solutions: List[str]) -> Dict:
        prompt = self.setup_prompt(query, ios)
        prompts = [prompt]
        resps = langmodels.openai_completion(
            self.engine,
            prompts=prompts,
            temperature=self.temperature,
            stop=["\n"],
            max_tokens=64,
            num_completions=self.num_repetitions,
        )

        attempts = 0
        while all(
            (resp.completions[0].text == "" or resp.completions[0].text.startswith("#"))
            for resp in resps
        ):
            if attempts == 2:
                break
            addition = "\n"
            prompts = [p + addition for p in prompts]
            resps = langmodels.openai_completion(
                self.engine,
                prompts=prompts,
                temperature=self.temperature,
                stop=["\n"],
                max_tokens=64,
            )
            attempts += 1

        for resp in resps:
            logger.debug(f"Query: {query}")
            logger.debug(resp.completions[0].text)
            eval_result = self.evaluate_result(resp.completions[0].text, ios)
            logger.debug(f"Eval Result: {eval_result}")
            if eval_result:
                return {
                    "success": True,
                    "query": query,
                    "generated_codes": [resp.completions[0].text],
                }

        return {
            "success": False,
            "query": query,
            "generated_codes": [resp.completions[0].text for resp in resps],
        }

    def run(self):
        if self.run_id is None:
            raise ValueError("run_id must be set")

        os.makedirs(self.results_save_path, exist_ok=True)
        path_to_res_file = os.path.join(
            self.results_save_path, f"codex_baseline_results_{self.run_id}.json"
        )
        path_to_log_file = os.path.join(
            self.results_save_path, f"codex_baseline_results_{self.run_id}.log"
        )
        logger.add(path_to_log_file, level="DEBUG")

        if os.path.exists(path_to_res_file):
            with open(path_to_res_file) as f:
                results = json.load(f)
        else:
            results = {}

        for name, benchmarks in [
            ("PandasEval1", load_benchmarks_1()),
            ("PandasEval2", load_benchmarks_2()),
        ]:
            bench_results: Dict[str, Dict] = results.get(name, {})
            results[name] = bench_results

            if self.ignore_eval_1 and name == "PandasEval1":
                continue
            if self.ignore_eval_2 and name == "PandasEval2":
                continue

            logger.debug(f"Running benchmark set {name}")
            logger.debug(f"Found {len(benchmarks)} benchmarks in {name}")

            for bench_id, benchmark in benchmarks.items():
                if benchmark.get("ignore", False):
                    continue

                single_bench_res = bench_results.get(bench_id, {})
                bench_results[bench_id] = single_bench_res

                logger.debug(f"Running benchmark {bench_id}")
                logger.debug(f"Number of Sets: {len(benchmark['sets'])}")
                for set_id, bench_set in benchmark["sets"].items():
                    for io in bench_set["ios"]:
                        assert len(io["inputs"]) == len(
                            io["invars"]
                        ), f"Failed for {bench_id} {set_id}"
                    logger.debug(f"Number of queries: {len(bench_set['queries'])}")
                    for idx, query in enumerate(bench_set["queries"]):
                        # if set_id != "D" or bench_id != "18":
                        # continue
                        key = f"{set_id}{idx}"
                        if (
                            not self.overwrite_existing_results
                        ) and key in single_bench_res:
                            logger.debug(
                                f"Skipping benchmark {bench_id}:{key} as result already exists"
                            )
                            continue
                        try:
                            res = self.run_benchmark(
                                query["query"], bench_set["ios"], bench_set["solutions"]
                            )
                            single_bench_res[key] = res
                            #  Dump results
                            with open(path_to_res_file, "w") as f:
                                json.dump(results, f, indent=4)

                        except Exception as e:
                            logger.debug(
                                f"Failed to run benchmark {bench_id} {set_id} {query['query']}"
                            )
                            logger.exception(e)
                            raise e


@attrs.define(repr=False, eq=False, slots=False)
class Algorithm1(CodexBaseline):
    codex_base_run_id: Optional[str] = None
    mining_campaign_dir: Optional[str] = None
    model_path: Optional[str] = None
    use_additional_descriptions: bool = True
    cross_encoder_path: Optional[str] = None

    @cached_property
    def searcher(self) -> EmbeddingBasedSearcher:
        if self.mining_campaign_dir is None:
            raise ValueError("mining_campaign_dir must be set")
        if self.model_path is None:
            raise ValueError("model_path must be set")

        return EmbeddingBasedSearcher(
            self.mining_campaign_dir,
            self.model_path,
            use_additional_descriptions=self.use_additional_descriptions,
        )

    def setup_prompt(self, query: str, ios: List[Dict], match: Dict):
        task_description: str = "#  Use the following code example as a guide to write code for the comment below"
        context = self.setup_context(ios[0])
        prompt = (
            f"{task_description}\n\n"
            f"#  {match['nl']}\n{match['code']}\n\n"
            f"{context}\n\n#  {query}\n"
        )

        return prompt

    def run_benchmark(self, query: str, ios: List[Dict], solutions: List[str]) -> Dict:
        db_matches = self.searcher.process_query(query, num_results=20)
        prompts = [self.setup_prompt(query, ios, match) for match in db_matches]
        resps = langmodels.openai_completion(
            self.engine, prompts=prompts, temperature=self.temperature, stop=["\n"]
        )

        trail: List[Dict[str, str]] = []
        for match, resp in zip(db_matches, resps):
            logger.debug(f"Query: {query}")
            logger.debug(resp.completions[0].text)
            eval_result = self.evaluate_result(resp.completions[0].text, ios)
            logger.debug(f"Eval Result: {eval_result}")
            trail.append(
                {
                    "ex_nl": match["nl"],
                    "ex_code": match["code"],
                    "generated_code": resp.completions[0].text,
                }
            )
            if eval_result:
                return {
                    "query": query,
                    "success": True,
                    "trail": trail,
                }

        return {
            "query": query,
            "success": False,
            "trail": trail,
        }

    def report(self):
        if self.run_id is None:
            raise ValueError("run_id must be set")

    def run(self):
        if self.codex_base_run_id is None:
            raise ValueError("codex_base_run_id must be set")

        os.makedirs(self.results_save_path, exist_ok=True)
        path_to_codex_res_file = os.path.join(
            self.results_save_path,
            f"codex_baseline_results_{self.codex_base_run_id}.json",
        )
        path_to_res_file = os.path.join(
            self.results_save_path, f"alg1_results_{self.run_id}.json"
        )
        path_to_log_file = os.path.join(
            self.results_save_path, f"alg1_results_{self.run_id}.log"
        )
        logger.add(path_to_log_file, level="DEBUG")

        if os.path.exists(path_to_res_file):
            with open(path_to_res_file) as f:
                results = json.load(f)
        else:
            results = {}

        if os.path.exists(path_to_codex_res_file):
            with open(path_to_codex_res_file) as f:
                codex_results = json.load(f)

        for name, benchmarks in [
            ("PandasEval1", load_benchmarks_1()),
            ("PandasEval2", load_benchmarks_2()),
        ]:
            bench_results: Dict[str, Dict] = results.get(name, {})
            codex_bench_results = codex_results.get(name, {})
            results[name] = bench_results

            if self.ignore_eval_1 and name == "PandasEval1":
                continue
            if self.ignore_eval_2 and name == "PandasEval2":
                continue

            logger.debug(f"Running benchmark set {name}")
            logger.debug(f"Found {len(benchmarks)} benchmarks in {name}")

            for bench_id, benchmark in benchmarks.items():
                if benchmark.get("ignore", False):
                    continue

                if bench_id not in codex_bench_results:
                    continue

                single_bench_res = bench_results.get(bench_id, {})
                bench_results[bench_id] = single_bench_res

                logger.debug(f"Running benchmark {bench_id}")
                logger.debug(f"Number of Sets: {len(benchmark['sets'])}")
                for set_id, bench_set in benchmark["sets"].items():
                    for io in bench_set["ios"]:
                        assert len(io["inputs"]) == len(
                            io["invars"]
                        ), f"Failed for {bench_id} {set_id}"
                    logger.debug(f"Number of queries: {len(bench_set['queries'])}")
                    for idx, query in enumerate(bench_set["queries"]):
                        key = f"{set_id}{idx}"
                        if key not in codex_bench_results[bench_id]:
                            continue
                        if codex_bench_results[bench_id][key]["success"]:
                            continue

                        if (
                            not self.overwrite_existing_results
                        ) and key in single_bench_res:
                            logger.debug(
                                f"Skipping benchmark {bench_id}:{key} as result already exists"
                            )
                            continue

                        try:
                            res = self.run_benchmark(
                                query["query"], bench_set["ios"], bench_set["solutions"]
                            )
                            single_bench_res[key] = {
                                **res,
                                "codex_results": codex_bench_results[bench_id][key],
                            }
                            #  Dump results
                            with open(path_to_res_file, "w") as f:
                                json.dump(results, f, indent=4)

                        except Exception as e:
                            logger.debug(
                                f"Failed to run benchmark {bench_id} {set_id} {query['query']}"
                            )
                            logger.exception(e)
                            raise e

                        # return


@attrs.define(repr=False, eq=False, slots=False)
class GenerationalModelBaseline:
    campaign_dir: str
    model_name: str = "Salesforce/codet5-base"

    @property
    def model_path(self) -> str:
        return os.path.join(self.campaign_dir, f"jigsaw_model_generational")

    @property
    def training_data_path(self):
        return os.path.join(self.model_path, f"training_data.pkl")

    @cached_property
    def non_derived_autodoc_results(self) -> List[AutodocResult]:
        success_path = "autodoc_success.pkl"
        autodoc_successes_path = os.path.join(self.campaign_dir, success_path)
        with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
            all_autodoc_results: List[AutodocResult] = list(
                res
                for res in tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
                if not res.is_derived
            )

        return all_autodoc_results

    @cached_property
    def training_data(self) -> pd.DataFrame:
        os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.training_data_path):
            return pickleutils.smart_load(self.training_data_path)

        all_autodoc_results = self.non_derived_autodoc_results
        records: List[Dict] = []
        seen: Set[str] = set()
        for res in tqdm.tqdm(all_autodoc_results, desc="Generating training data"):
            assert isinstance(res, AutodocResult)
            for desc in res.canonical_descs:
                text = desc.desc.primary_desc
                if text not in seen:
                    records.append(
                        {
                            "source_text": f"generate-code: {text}",
                            "target_text": desc.target_code,
                        }
                    )
                    seen.add(text)

                text = desc.desc.primary_desc.replace('"', "'")
                if text not in seen:
                    records.append(
                        {
                            "source_text": f"generate-code: {text}",
                            "target_text": desc.target_code,
                        }
                    )
                    seen.add(text)

                text = desc.desc.primary_desc.replace('"', "")
                if text not in seen:
                    records.append(
                        {
                            "source_text": f"generate-code: {text}",
                            "target_text": desc.target_code,
                        }
                    )
                    seen.add(text)

        logger.debug(f"Generated {len(records)} training data points")
        random.shuffle(records)

        result = pd.DataFrame.from_records(records)
        pickleutils.smart_dump(result, self.training_data_path)

        return result

    def train_generational_model(self, max_epochs: int = 10):
        os.makedirs(self.model_path, exist_ok=True)

        df = self.training_data
        # for i, j in df.head().values:
        #     print(i, j)
        # return
        # df = df.iloc[:100, :]
        train_df, test_df = train_test_split(df, test_size=0.2)

        if self.model_name.startswith("Salesforce/codet5"):
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        else:
            tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, return_dict=True
        )

        simplet5_model = SimpleT5()
        simplet5_model.tokenizer = tokenizer
        simplet5_model.model = model

        output_dir = os.path.join(self.model_path, "training_outputs")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

        simplet5_model.train(
            train_df=train_df,
            eval_df=test_df,
            source_max_token_len=128,
            target_max_token_len=128,
            outputdir=output_dir,
            early_stopping_patience_epochs=3,
            batch_size=8,
            max_epochs=max_epochs,
            use_gpu=True,
        )

        simplet5_model.model.save_pretrained(
            os.path.join(self.model_path, "trained_model")
        )
        simplet5_model.tokenizer.save_pretrained(
            os.path.join(self.model_path, "trained_model")
        )

    @cached_property
    def model(self) -> SimpleT5:
        model = SimpleT5()
        path = os.path.join(self.model_path, "trained_model")
        if self.model_name.startswith("Salesforce/codet5"):
            model.tokenizer = RobertaTokenizer.from_pretrained(path)
        else:
            model.tokenizer = T5Tokenizer.from_pretrained(path)

        model.model = T5ForConditionalGeneration.from_pretrained(path)
        model.device = torch.device("cuda")
        model.model = model.model.to(model.device)
        return model

    def predict(self, query: str, num_results: int = 20) -> List[str]:
        model: SimpleT5 = self.model

        predicted = model.predict(
            query, num_return_sequences=num_results, num_beams=num_results * 2
        )

        return predicted


@attrs.define(repr=False, eq=False, slots=False)
class GenerationalModelWithSearchBaseline(GenerationalModelBaseline):
    campaign_dir: str
    model_name: str = "Salesforce/codet5-base"

    @property
    def model_path(self) -> str:
        return os.path.join(self.campaign_dir, f"jigsaw_model_generational_w_search")

    @property
    def training_data_path(self):
        return os.path.join(self.model_path, f"training_data.pkl")

    @cached_property
    def training_data(self) -> pd.DataFrame:
        os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.training_data_path):
            return pickleutils.smart_load(self.training_data_path)

        all_autodoc_results = self.non_derived_autodoc_results
        records: List[Dict] = []
        seen: Set[str] = set()
        for res in tqdm.tqdm(all_autodoc_results, desc="Generating training data"):
            assert isinstance(res, AutodocResult)
            param_pool: List[Dict] = [
                {
                    "nl": desc.parameterized_nl,
                    "code": desc.parameterized_code,
                }
                for desc in res.canonical_descs
            ]

            for desc in res.canonical_descs:
                text = desc.desc.primary_desc
                if text not in seen:
                    param = random.choice(param_pool)
                    records.append(
                        {
                            "source_text": (
                                f"ParamNL: {param['nl']}\n"
                                f"ParamCode: {param['code']}\n"
                                f"Target Text: {text}\n"
                            ),
                            "target_text": desc.target_code,
                        }
                    )
                    seen.add(text)

                text = desc.desc.primary_desc.replace('"', "'")
                if text not in seen:
                    param = random.choice(param_pool)
                    records.append(
                        {
                            "source_text": (
                                f"ParamNL: {param['nl']}\n"
                                f"ParamCode: {param['code']}\n"
                                f"Target Text: {text}\n"
                            ),
                            "target_text": desc.target_code,
                        }
                    )
                    seen.add(text)

                text = desc.desc.primary_desc.replace('"', "")
                if text not in seen:
                    param = random.choice(param_pool)
                    records.append(
                        {
                            "source_text": (
                                f"ParamNL: {param['nl']}\n"
                                f"ParamCode: {param['code']}\n"
                                f"Target Text: {text}\n"
                            ),
                            "target_text": desc.target_code,
                        }
                    )
                    seen.add(text)

        logger.debug(f"Generated {len(records)} training data points")
        random.shuffle(records)

        result = pd.DataFrame.from_records(records)
        pickleutils.smart_dump(result, self.training_data_path)

        return result

    def predict(self, query: str, matches: List[Dict]) -> List[str]:
        model: SimpleT5 = self.model

        results: List[str] = []

        for match in matches:
            prompt: str = (
                f"ParamNL: {match['nl']}\n"
                f"ParamCode: {match['code']}\n"
                f"Target Text: {query}\n"
            )
            prediction = model.predict(prompt, num_return_sequences=1, num_beams=2)[0]
            print("WOW", prediction)
            results.append(prediction)

        return results


@attrs.define(repr=False, eq=False, slots=False)
class Algorithm2(CodexBaseline):
    run_id: Optional[str] = None
    mining_campaign_dir: Optional[str] = None
    model_path: Optional[str] = None
    use_additional_descriptions: bool = True
    instantiation_model_type: str = "generational"
    instantiation_model_path: Optional[str] = None

    @cached_property
    def searcher(self) -> EmbeddingBasedSearcher:
        if self.mining_campaign_dir is None:
            raise ValueError("mining_campaign_dir must be set")
        if self.model_path is None:
            raise ValueError("model_path must be set")

        return EmbeddingBasedSearcher(
            self.mining_campaign_dir,
            self.model_path,
            use_additional_descriptions=self.use_additional_descriptions,
        )

    @cached_property
    def instantiator(
        self,
    ) -> Union[GenerationalModelBaseline, GenerationalModelWithSearchBaseline]:
        if self.instantiation_model_path is None:
            raise ValueError("instantiation_model_path must be set")

        if self.instantiation_model_type == "generational":
            return GenerationalModelBaseline(self.mining_campaign_dir)
        elif self.instantiation_model_type == "generational_w_search":
            return GenerationalModelWithSearchBaseline(self.mining_campaign_dir)
        else:
            raise ValueError(
                f"Unknown instantiation_model_type: {self.instantiation_model_type}"
            )

    def setup_prompt(self, query: str, ios: List[Dict], match: Dict):
        task_description: str = "#  Use the following code example as a guide to write code for the comment below"
        context = self.setup_context(ios[0])
        prompt = (
            f"{task_description}\n\n"
            f"#  {match['nl']}\n{match['code']}\n\n"
            f"{context}\n\n#  {query}\n"
        )

        return prompt

    def prepare_progs(self, match: Dict, ios: List[Dict]) -> List[str]:
        ctx_state = {}
        imports = "\n".join(
            [
                "import pandas",
                "import pandas as pd",
                "import numpy",
                "import numpy as np",
            ]
        )
        exec(imports, ctx_state)
        exec(self.setup_context(ios[0]), ctx_state)
        logger.debug(f"Okay: {match['param_code']}")
        return []

    def run_benchmark(self, query: str, ios: List[Dict], solutions: List[str]) -> Dict:
        logger.debug("RUNNING!")
        if self.instantiation_model_type == "generational":
            db_matches = [{"nl": "N/A", "code": "N/A"} for _ in range(20)]
            assert isinstance(self.instantiator, GenerationalModelBaseline)
            predictions = self.instantiator.predict(query, num_results=20)
        elif self.instantiation_model_type == "generational_w_search":
            assert isinstance(self.instantiator, GenerationalModelWithSearchBaseline)
            db_matches = self.searcher.process_query(query, num_results=20)
            predictions = self.instantiator.predict(query, db_matches)
        else:
            raise ValueError(
                f"Unknown instantiation_model_type: {self.instantiation_model_type}"
            )

        trail: List[Dict[str, str]] = []
        for match, pred in zip(db_matches, predictions):
            logger.debug(f"Query: {query}")
            logger.debug(f"Pred: {pred}")
            eval_result = self.evaluate_result(pred, ios)
            logger.debug(f"Eval Result: {eval_result}")
            trail.append(
                {
                    "ex_nl": match["nl"],
                    "ex_code": match["code"],
                    "generated_code": pred,
                }
            )
            if eval_result:
                return {
                    "query": query,
                    "success": True,
                    "trail": trail,
                }

        return {
            "query": query,
            "success": False,
            "trail": trail,
        }

    def report(self):
        if self.run_id is None:
            raise ValueError("run_id must be set")

    def run(self):
        if self.run_id is None:
            raise ValueError("run_id must be set")

        os.makedirs(self.results_save_path, exist_ok=True)
        path_to_codex_res_file = os.path.join(
            self.results_save_path, "codex_baseline_results.json"
        )
        path_to_res_file = os.path.join(
            self.results_save_path, f"alg1_results_{self.run_id}.json"
        )
        path_to_log_file = os.path.join(
            self.results_save_path, f"alg1_results_{self.run_id}.log"
        )
        logger.add(path_to_log_file, level="DEBUG")

        if os.path.exists(path_to_res_file):
            with open(path_to_res_file) as f:
                results = json.load(f)
        else:
            results = {}

        if os.path.exists(path_to_codex_res_file):
            with open(path_to_codex_res_file) as f:
                codex_results = json.load(f)

        for name, benchmarks in [
            ("PandasEval1", load_benchmarks_1()),
            ("PandasEval2", load_benchmarks_2()),
        ]:
            bench_results: Dict[str, Dict] = results.get(name, {})
            codex_bench_results = codex_results.get(name, {})
            results[name] = bench_results

            if self.ignore_eval_1 and name == "PandasEval1":
                continue
            if self.ignore_eval_2 and name == "PandasEval2":
                continue

            logger.debug(f"Running benchmark set {name}")
            logger.debug(f"Found {len(benchmarks)} benchmarks in {name}")

            for bench_id, benchmark in benchmarks.items():
                if benchmark.get("ignore", False):
                    continue

                if bench_id not in codex_bench_results:
                    continue

                if (not self.overwrite_existing_results) and bench_id in bench_results:
                    logger.debug(
                        f"Skipping benchmark {bench_id} as result already exists"
                    )
                    continue

                bench_results[bench_id] = {}

                logger.debug(f"Running benchmark {bench_id}")
                logger.debug(f"Number of Sets: {len(benchmark['sets'])}")
                for set_id, bench_set in benchmark["sets"].items():
                    for io in bench_set["ios"]:
                        assert len(io["inputs"]) == len(
                            io["invars"]
                        ), f"Failed for {bench_id} {set_id}"
                    logger.debug(f"Number of queries: {len(bench_set['queries'])}")
                    for idx, query in enumerate(bench_set["queries"]):
                        key = f"{set_id}{idx}"
                        if key not in codex_bench_results[bench_id]:
                            continue
                        # if codex_bench_results[bench_id][key]["success"]:
                        # continue

                        try:
                            res = self.run_benchmark(
                                query["query"], bench_set["ios"], bench_set["solutions"]
                            )
                            bench_results[bench_id][key] = {
                                **res,
                                "codex_results": codex_bench_results[bench_id][key],
                            }
                            #  Dump results
                            with open(path_to_res_file, "w") as f:
                                json.dump(results, f, indent=4)

                        except Exception as e:
                            logger.debug(
                                f"Failed to run benchmark {bench_id} {set_id} {query['query']}"
                            )
                            logger.exception(e)
                            raise e


if __name__ == "__main__":
    fire.Fire()
