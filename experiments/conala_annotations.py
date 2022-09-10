import json
import os
import sys
from typing import Dict, List

import attrs
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, flash, request, redirect, url_for

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)

DATA: "ConalaData"
PAGE_SIZE: int = 10


@attrs.define(eq=False, repr=False, slots=False)
class ConalaData:
    campaign_dir: str
    raw_data: List[Dict]

    _raw_data_by_key: Dict[str, Dict] = attrs.field(init=False, factory=dict)

    def __attrs_post_init__(self):
        self._raw_data_by_key = {d["id"]: d for d in self.raw_data}

    @classmethod
    def from_campaign_dir(cls, campaign_dir: str) -> "ConalaData":
        processed_path = os.path.join(campaign_dir, "conala_pandas_processed.json")
        if not os.path.exists(processed_path):
            with open(processed_path, "w") as f_w, open(
                os.path.join(campaign_dir, "conala_pandas_full.json")
            ) as f_r:
                f_w.write(f_r.read())

        with open(processed_path, "r") as f:
            raw_data = json.load(f)

        assert isinstance(raw_data, list)
        assert isinstance(raw_data[0], dict)
        raw_data = [i for i in raw_data if i["id"].endswith("_0")]
        for idx, item in enumerate(raw_data):
            if "status" not in item:
                item["status"] = "Unprocessed"
                item["metadata"] = {}

            item["idx"] = idx

        result = ConalaData(campaign_dir, raw_data)
        result.save()
        return result

    @property
    def items(self) -> List[Dict]:
        return self.raw_data

    @property
    def num_pages(self) -> int:
        return (len(self.raw_data) + PAGE_SIZE - 1) // PAGE_SIZE

    def __getitem__(self, item):
        return self._raw_data_by_key[item]

    def save(self):
        with open(
            os.path.join(self.campaign_dir, "conala_pandas_processed.json"), "w"
        ) as f:
            json.dump(self.raw_data, f, indent=2)

    def get_num_processed(self) -> int:
        return sum(1 for i in self.items if i["status"] == "Processed")

    def get_num_annotated(self) -> int:
        return sum(1 for i in self.items if i["status"] != "Unprocessed")


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
    page_items = DATA.items[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    return render_template(
        "index_annotations.html",
        page=page,
        page_items=page_items,
        max_pages=DATA.num_pages,
    )


@app.route("/process/<string:item_id>", methods=["GET", "POST"])
def process(item_id: str):
    item = DATA[item_id]
    page_num = item["idx"] // PAGE_SIZE
    prev_id = item["id"] if item["idx"] == 0 else DATA.items[item["idx"] - 1]["id"]
    next_id = (
        item["id"]
        if item["idx"] == len(DATA.items) - 1
        else DATA.items[item["idx"] + 1]["id"]
    )

    if request.method == "GET":
        return render_template(
            "process_annotations.html",
            item=item,
            page_num=page_num,
            prev_id=prev_id,
            next_id=next_id,
            num_processed=DATA.get_num_processed(),
            num_annotated=DATA.get_num_annotated(),
        )

    elif request.method == "POST":
        if request.form["status_selection"] == "deletion":
            item["status"] = "Deleted"
            item["metadata"].clear()
        elif request.form["status_selection"] == "reset":
            item["status"] = "Unprocessed"
            item["metadata"].clear()
        else:
            item["status"] = "Processed"
            item["metadata"]["revised_intent"] = request.form["revised_intent"]
            item["metadata"]["code_context"] = request.form["code_context"]
            item["metadata"]["revised_gt"] = request.form["revised_gt"]
            item["metadata"]["abstract_gt"] = request.form["abstract_gt"]

        DATA.save()
        return redirect(url_for("process", item_id=item_id, just_saved=1))


@app.route("/stackoverflow/<string:question_id>")
def stackoverflow(question_id: str):
    return get_stackoverflow_html(question_id)


if __name__ == "__main__":
    campaign_dir = sys.argv[1]
    DATA = ConalaData.from_campaign_dir(campaign_dir)
    app.run()
