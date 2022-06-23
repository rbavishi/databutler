import os
import random
from contextlib import contextmanager
from typing import List, cast, Dict, Optional

import requests
import json
import fire
import tqdm

from databutler.utils import pickleutils, multiprocess


def fetch_notebooks_list(campaign_dir: str) -> None:
    os.makedirs(campaign_dir, exist_ok=True)
    url = "https://github-notebooks-samples.s3-eu-west-1.amazonaws.com/ntbs_list.json"
    print(f"Fetching notebooks list from {url}")
    response = requests.get(url)
    data = json.loads(response.text)
    with open(os.path.join(campaign_dir, "notebooks_list.json"), "w") as f:
        json.dump(data, f)

    print(f"Saved {len(data)} notebooks to {campaign_dir}")


def get_nb(nb_key: str) -> Optional[Dict]:
    url = f"https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/{nb_key}"
    response = requests.get(url)
    text = response.text
    if any(snippet in text for snippet in ["import pandas", "from pandas"]):
        nb_json = json.loads(text)
        if "cells" not in nb_json:
            return None
        nb_json["cells"] = [
            cell for cell in nb_json["cells"]
            if "source" in cell and cell["cell_type"] == "code"
        ]
        for cell in nb_json["cells"]:
            if "metadata" in cell:
                cell["metadata"] = {"trusted": True}
            if "outputs" in cell:
                cell["outputs"] = []

        return {"nb_key": nb_key, "json": nb_json}
    else:
        return None


@contextmanager
def get_nb_reader(campaign_dir: str) -> pickleutils.PickledMapReader:
    save_path = os.path.join(campaign_dir, "downloaded_notebooks.pkl")
    with pickleutils.PickledMapReader(save_path) as reader:
        yield reader


def download_notebooks(campaign_dir: str, save_frequency: int = 1000, num_processes: int = 2, num_notebooks: Optional[int] = None) -> None:
    os.makedirs(campaign_dir, exist_ok=True)
    nb_list_path = os.path.join(campaign_dir, "notebooks_list.json")
    if not os.path.exists(nb_list_path):
        fetch_notebooks_list(campaign_dir)

    with open(nb_list_path, "r") as f:
        nb_list = json.load(f)

    assert isinstance(nb_list, list)
    assert len(nb_list) > 0
    assert isinstance(nb_list[0], str)

    if num_notebooks is not None:
        nb_list = nb_list[:num_notebooks]
        print(f"Only downloading {num_notebooks} notebooks.")
    else:
        print(f"Downloading {len(nb_list)} notebooks")

    save_path = os.path.join(campaign_dir, "downloaded_notebooks.pkl")
    skipped_path = os.path.join(campaign_dir, "skipped_notebooks.pkl")
    with pickleutils.PickledMapWriter(save_path, overwrite_existing=False) as writer, pickleutils.PickledMapWriter(skipped_path, overwrite_existing=False) as skipped_writer:
        print(f"Already downloaded {len(writer)}, and skipped {len(skipped_writer)} notebooks")
        to_process: List[str] = cast(List[str], list(set(nb_list) - set(writer.keys()) - set(skipped_writer.keys())))
        print(f"Processing {len(to_process)} notebooks")

        random.shuffle(to_process)

        ctr = 0
        succ = skipped = failures = 0
        with tqdm.tqdm(dynamic_ncols=True, desc="Downloading", total=len(to_process)) as pbar:
            for idx in range(0, len(to_process), 10000):
                chunk = to_process[idx: idx + 10000]
                for nb_key, res in zip(chunk, multiprocess.run_tasks_in_parallel_iter(get_nb, chunk, num_workers=num_processes)):
                    if res.is_success():
                        if res.result is None:
                            skipped += 1
                            skipped_writer[nb_key] = False
                        else:
                            try:
                                writer[res.result["nb_key"]] = res.result["json"]
                                succ += 1
                            except:
                                failures += 1
                    else:
                        failures += 1

                    pbar.set_postfix(success=succ, skipped=skipped, failures=failures)
                    pbar.update()

                    ctr += 1
                    if ctr % save_frequency == 0:
                        writer.flush()
                        skipped_writer.flush()

        writer.flush()
        skipped_writer.flush()


if __name__ == "__main__":
    fire.Fire({
        "download_notebooks": download_notebooks,
    })
