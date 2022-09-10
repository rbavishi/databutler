import hashlib
import os
import sys
from typing import Dict, List

import attrs
import click
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for

from databutler.mining.static_pandas_mining.search import run_search_engine
from databutler.utils import pickleutils
from experiments.conala_annotations import ConalaData

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

PAGE_SIZE: int = 10


@attrs.define(eq=False, repr=False)
class InspectionItem:
    id: str
    question_id: str
    intent: str
    ground_truth: str
    our_results: List[Dict]
    copilot_result: str
    status: str
    metadata: Dict = attrs.field(factory=dict)


INSPECTION_DATA: List[InspectionItem] = []


def run_search_ours(
    campaign_dir: str, identifier: str, model_path: str, data: ConalaData
) -> List[List[Dict]]:
    items = [i for i in data.items if i["status"] == "Processed"]
    queries = [i["metadata"]["revised_intent"] for i in items]

    return run_search_engine(model_path, queries)


def prepare_data_for_inspection(
    campaign_dir: str, identifier: str, model_path: str
) -> List[InspectionItem]:
    data = ConalaData.from_campaign_dir(campaign_dir)
    items = [i for i in data.items if i["status"] == "Processed"]
    hash_value = hashlib.sha256(
        str((campaign_dir, identifier, model_path, repr(items))).encode("utf-8")
    ).hexdigest()
    cache_path = os.path.join(campaign_dir, f".inspection_data_cache_{hash_value}.pkl")
    save_path = os.path.join(campaign_dir, f"inspection_data_saved.pkl")
    if os.path.exists(cache_path):
        if os.path.exists(save_path):
            return pickleutils.smart_load(save_path)
        else:
            return pickleutils.smart_load(cache_path)

    our_results = run_search_ours(campaign_dir, identifier, model_path, data)
    copilot_results: List[str] = []
    for idx, item in enumerate(items):
        copilot_result_path = os.path.join(
            campaign_dir, identifier, "copilot", f"copilot_{idx}.py"
        )
        with open(copilot_result_path, "r") as f:
            copilot_results.append(f.read())

    inspection_items: List[InspectionItem] = [
        InspectionItem(
            id=item["id"],
            question_id=item["question_id"],
            intent=item["metadata"]["revised_intent"],
            ground_truth=item["metadata"]["revised_gt"],
            our_results=our_results[idx],
            copilot_result=copilot_results[idx],
            status="Unprocessed",
            metadata={},
        )
        for idx, item in enumerate(items)
    ]

    pickleutils.smart_dump(inspection_items, cache_path)
    if os.path.exists(save_path):
        if click.confirm(f"Do you want to overwrite the saved inspection data?"):
            pickleutils.smart_dump(inspection_items, save_path)
        else:
            print(f"Cancelling")
            sys.exit(0)

    return inspection_items


def get_stackoverflow_html(question_id: str) -> str:
    url = f"https://stackoverflow.com/questions/{question_id}"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    for s in soup.select("script"):
        s.extract()

    for s in soup.select("div", {"class": "js-consent-banner"}):
        if "cookies" in s.text:
            s.extract()

    return str(soup)


@app.route("/")
@app.route("/<int:page>")
def index(page: int = 0):
    page_items = INSPECTION_DATA[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    max_pages = (len(INSPECTION_DATA) + (PAGE_SIZE - 1)) // PAGE_SIZE
    return render_template(
        "index_comparison.html", page=page, page_items=page_items, max_pages=max_pages
    )


@app.route("/process/<string:item_id>", methods=["GET", "POST"])
def process(item_id: str):
    idx, item = next(
        (idx, i) for idx, i in enumerate(INSPECTION_DATA) if i.id == item_id
    )
    page_num = idx // PAGE_SIZE
    prev_id = item.id if idx == 0 else INSPECTION_DATA[idx - 1].id
    next_id = (
        item.id if idx == (len(INSPECTION_DATA) - 1) else INSPECTION_DATA[idx + 1].id
    )

    if request.method == "GET":
        return render_template(
            "process_comparison.html",
            item=item,
            page_num=page_num,
            prev_id=prev_id,
            next_id=next_id,
            num_processed=sum(1 for i in INSPECTION_DATA if i.status == "Processed"),
        )

    elif request.method == "POST":
        item.status = "Processed"
        item.metadata["copilot_verdict"] = request.form["copilot_verdict"]
        item.metadata["ours_verdict"] = request.form["ours_verdict"]
        item.metadata["win_verdict"] = request.form["win_verdict"]
        item.metadata["pres_in_corpus"] = request.form["pres_in_corpus"]

        save_path = os.path.join(campaign_dir, f"inspection_data_saved.pkl")
        pickleutils.smart_dump(INSPECTION_DATA, save_path)
        return redirect(url_for("process", item_id=item_id, just_saved=1))


@app.route("/stackoverflow/<string:question_id>")
def stackoverflow(question_id: str):
    return get_stackoverflow_html(question_id)


@app.route("/analysis")
def analysis():
    df = pd.DataFrame.from_records(
        [i.metadata for i in INSPECTION_DATA if i.status == "Processed"]
    )
    df["copilot_verdict"] = df["copilot_verdict"].apply(lambda x: 1 if x == "3" else 0)
    df["ours_verdict"] = df["ours_verdict"].apply(lambda x: 1 if x == "3" else 0)
    df["ours_best"] = [
        1 if row[1].ours_verdict == "3" or row[1].pres_in_corpus == "1" else 0
        for row in df.iterrows()
    ]
    cross_cur = pd.crosstab(df["copilot_verdict"], df["ours_verdict"])
    cross_best = pd.crosstab(df["copilot_verdict"], df["ours_best"])
    return cross_cur.to_html() + cross_best.to_html()


if __name__ == "__main__":
    campaign_dir, identifier, model_path = sys.argv[1:]
    INSPECTION_DATA = prepare_data_for_inspection(campaign_dir, identifier, model_path)

    app.run(port=5050)
