import os

import fire

from databutler.utils import code as codeutils
from experiments.conala_annotations import ConalaData


def setup_copilot(campaign_dir: str, identifier: str) -> None:
    path = os.path.join(campaign_dir, identifier, "copilot")
    if os.path.exists(path):
        print(f"{path} already exists")
        return

    os.makedirs(path, exist_ok=True)
    data = ConalaData.from_campaign_dir(campaign_dir)
    items = [i for i in data.items if i["status"] == "Processed"]
    print(f"Found {len(items)} annotated items")

    for idx, item in enumerate(items):
        filepath = os.path.join(path, f"copilot_{idx}.py")
        context = codeutils.normalize_code_fast(item["metadata"]["code_context"])
        intent = item["metadata"]["revised_intent"]
        with open(filepath, "w") as f:
            f.write(f"{context.rstrip()}\n\n#  {intent}")


if __name__ == "__main__":
    fire.Fire()
