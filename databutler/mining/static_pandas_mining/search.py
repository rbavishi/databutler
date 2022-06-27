import collections
import itertools
import os
import random
import shutil
from typing import List, Dict, Iterator, Set, Tuple

import fire
import numpy as np
import torch
import tqdm

import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    util,
)
from simplet5 import SimpleT5
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration

from databutler.mining.static_pandas_mining.autodoc import AUTODOC_SUCCESSES_PATH
from databutler.mining.static_pandas_mining.autodoc_utils import (
    AutodocResult,
    AutodocDescription,
)
from databutler.utils import pickleutils


def prepare_dataset_for_generational_model(
    campaign_dir: str, num_per_desc_uid: int = 10
):
    """Prepare a dataset out of the autodoc results to be used for training a small model like CodeBERT or CodeT5"""
    autodoc_successes_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
        all_autodoc_results: List[AutodocResult] = list(
            tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
        )

    print(f"Found {len(all_autodoc_results)} autodoc results")
    descriptions_by_uid: Dict[str, List[AutodocDescription]] = collections.defaultdict(
        list
    )
    for res in all_autodoc_results:
        for desc in res.correct_descriptions:
            descriptions_by_uid[desc.uid].append(desc)

    #  Shuffle each array
    for descs in descriptions_by_uid.values():
        random.shuffle(descs)

    print(f"Found {len(descriptions_by_uid)} unique descriptions")

    iter_dict: Dict[str, Iterator[AutodocDescription]] = {
        k: itertools.cycle(v) for k, v in descriptions_by_uid.items()
    }

    records = []
    for loop_no in range(0, num_per_desc_uid):
        for iter_ in iter_dict.values():
            desc = next(iter_)
            nl = desc.nl
            code = desc.generated_code
            if code.startswith("def "):
                code = "\n".join(code.split("\n")[1:])
            if code.startswith("return "):
                code = code[len("return ") :]

            source_text = f"generate-code: {nl}"
            target_text = code

            records.append({"source_text": source_text, "target_text": target_text})

    df = pd.DataFrame.from_records(records)
    print(df)
    pickleutils.smart_dump(df, os.path.join(campaign_dir, "t5_code_data.pkl"))


def get_max_tokens(tokenizer, strings: List[str]) -> int:
    lengths = []
    for idx in range(0, len(strings), 32):
        batch = strings[idx : idx + 32]
        lengths.extend(len(i) for i in tokenizer(batch)["input_ids"])

    print(
        np.mean(lengths),
        np.median(lengths),
        np.max(lengths),
        np.min(lengths),
        np.percentile(lengths, 50),
        np.percentile(lengths, 75),
        np.percentile(lengths, 90),
        np.percentile(lengths, 99),
    )
    return max(lengths)


def train_generational_model(campaign_dir: str, model_name: str, max_epochs: int = 10):
    df = pickleutils.smart_load(os.path.join(campaign_dir, "t5_code_data.pkl"))
    train_df, test_df = train_test_split(df, test_size=0.2)

    if model_name.startswith("Salesforce/codet5"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

    simplet5_model = SimpleT5()
    simplet5_model.tokenizer = tokenizer
    simplet5_model.model = model

    prefix = model_name.replace("/", "_")

    output_dir = os.path.join(campaign_dir, prefix + "_model_outputs")
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
        os.path.join(campaign_dir, prefix + "_trained")
    )
    simplet5_model.tokenizer.save_pretrained(
        os.path.join(campaign_dir, prefix + "_trained")
    )


def prepare_dataset_for_training_embeddings(
    campaign_dir: str, per_equiv_class: int = 10
):
    autodoc_successes_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
        all_autodoc_results: List[AutodocResult] = list(
            tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
        )

    desc_by_uids: Dict[str, List[AutodocDescription]] = collections.defaultdict(list)
    for res in all_autodoc_results:
        for desc in res.correct_descriptions:
            if desc.is_derived:
                desc_by_uids[desc.uid].append(desc)

    non_derived_results: List[AutodocResult] = [
        res
        for res in all_autodoc_results
        if all(not desc.is_derived for desc in res.correct_descriptions)
    ]
    print(f"Found {len(non_derived_results)} non-derived autodoc results")

    equiv_classes: List[List[str]] = []
    for res in non_derived_results:
        unique_nls: Set[str] = set()
        for desc in res.correct_descriptions:
            unique_nls.add(desc.nl)
            if desc.llm_based_parameterization is not None:
                unique_nls.add(desc.llm_based_parameterization.nl)

            if len(desc_by_uids[desc.uid]) >= 1:
                unique_nls.add(random.choice(desc_by_uids[desc.uid]).nl)

        unique_nls.update(desc.nl for desc in res.incorrect_descriptions)
        equiv_classes.append(list(unique_nls))
        random.shuffle(equiv_classes[-1])

    all_indices: List[int] = list(range(len(equiv_classes)))
    correct_pairs: List[Tuple[str, str]] = []
    incorrect_pairs: List[Tuple[str, str]] = []
    for idx, equiv_class in enumerate(tqdm.tqdm(equiv_classes, desc="Preparing data")):
        candidates = list(
            itertools.islice(itertools.combinations(equiv_class, 2), per_equiv_class)
        )
        random.shuffle(candidates)

        for s1, s2 in candidates:
            correct_pairs.append((s1, s2))

        other_sample = random.sample(all_indices, per_equiv_class)
        while idx in other_sample:
            other_sample = random.sample(all_indices, per_equiv_class)

        for other_idx in other_sample:
            s1 = random.choice(equiv_class)
            s2 = random.choice(equiv_classes[other_idx])
            incorrect_pairs.append((s1, s2))

    print("SIZES", len(correct_pairs), len(incorrect_pairs))
    pickleutils.smart_dump(
        [correct_pairs, incorrect_pairs],
        os.path.join(campaign_dir, "embedding_train_data.pkl"),
    )


def train_embeddings(
    campaign_dir: str, loss_type: str = "contrastive", num_epochs: int = 5
):
    correct_pairs, incorrect_pairs = pickleutils.smart_load(
        os.path.join(campaign_dir, "embedding_train_data.pkl")
    )
    random.shuffle(correct_pairs)
    random.shuffle(incorrect_pairs)

    model = SentenceTransformer("all-mpnet-base-v2")
    train_examples = [
        *(
            InputExample(texts=[s1, s2], label=1)
            for s1, s2 in correct_pairs[: int(len(correct_pairs) * 0.8)]
        ),
        *(
            InputExample(texts=[s1, s2], label=0)
            for s1, s2 in incorrect_pairs[: int(len(incorrect_pairs) * 0.8)]
        ),
    ]

    test_examples = [
        *(
            InputExample(texts=[s1, s2], label=1)
            for s1, s2 in correct_pairs[int(len(correct_pairs) * 0.8) :]
        ),
        *(
            InputExample(texts=[s1, s2], label=0)
            for s1, s2 in incorrect_pairs[int(len(incorrect_pairs) * 0.8) :]
        ),
    ]

    if loss_type == "contrastive":
        train_loss = losses.ContrastiveLoss(model=model)
        output_path = os.path.join(campaign_dir, "embedding_model_contrastive")
    elif loss_type == "cosine":
        train_loss = losses.CosineSimilarityLoss(model=model)
        output_path = os.path.join(campaign_dir, "embedding_model_cosine")
        for ex in itertools.chain(train_examples, test_examples):
            ex.label = float(ex.label)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        test_examples, show_progress_bar=True
    )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

    model.fit(
        [(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=10000,
        show_progress_bar=True,
        save_best_model=True,
        output_path=output_path,
    )


def prepare_search_engine(campaign_dir: str, model_path: str):
    model = SentenceTransformer(model_path)

    autodoc_successes_path = os.path.join(campaign_dir, AUTODOC_SUCCESSES_PATH)
    with pickleutils.PickledMapReader(autodoc_successes_path) as autodoc_reader:
        all_autodoc_results: List[AutodocResult] = list(
            tqdm.tqdm(autodoc_reader.values(), total=len(autodoc_reader))
        )

    corpus: List[AutodocDescription] = []
    for res in all_autodoc_results:
        for desc in res.correct_descriptions:
            if desc.is_derived:
                continue
            corpus.append(desc)

    print(f"Found {len(corpus)} autodoc descriptions")

    embeddings = model.encode(
        [desc.nl for desc in corpus], show_progress_bar=True, convert_to_tensor=True
    )
    print("Finished generating embeddings")

    embeddings_path = os.path.join(model_path, "search_engine_embeddings.pkl")
    pickleutils.smart_dump((corpus, embeddings), embeddings_path)
    print(f"Saved corpus and embeddings to {embeddings_path}")


def start_search_engine(campaign_dir: str, model_path: str):
    corpus, embeddings = pickleutils.smart_load(
        os.path.join(model_path, "search_engine_embeddings.pkl")
    )
    print(f"Loaded corpus and embeddings from {model_path}")

    model = SentenceTransformer(model_path)
    while True:
        query = input("Query: ")
        query_embedding = model.encode(
            query, show_progress_bar=False, convert_to_tensor=True
        )
        distances = util.cos_sim(query_embedding, embeddings)[0]

        top_results = torch.topk(distances, k=100)
        seen_code: Set[str] = set()
        ctr = 0
        for score, idx in zip(top_results[0], top_results[1]):
            if corpus[idx].generated_code in seen_code:
                continue

            seen_code.add(corpus[idx].generated_code)
            print(corpus[idx].nl, "(Score: {:.4f})".format(score))
            print(corpus[idx].generated_code)
            print("---")
            ctr += 1

            if ctr == 10:
                break

        print("\n----------\n")


def run_search_engine(model_path: str, queries: List[str]):
    corpus, embeddings = pickleutils.smart_load(
        os.path.join(model_path, "search_engine_embeddings.pkl")
    )
    print(f"Loaded corpus and embeddings from {model_path}")

    model = SentenceTransformer(model_path)
    all_results: List[List[Dict]] = []
    for query in tqdm.tqdm(queries, desc="Processing queries"):
        query_embedding = model.encode(
            query, show_progress_bar=False, convert_to_tensor=True
        )
        distances = util.cos_sim(query_embedding, embeddings)[0]

        top_results = torch.topk(distances, k=100)
        seen_code: Set[str] = set()
        ctr = 0
        results: List[Dict] = []
        for score, idx in zip(top_results[0], top_results[1]):
            if corpus[idx].generated_code in seen_code:
                continue

            seen_code.add(corpus[idx].generated_code)
            results.append(
                {
                    "score": score,
                    "nl": corpus[idx].nl,
                    "code": corpus[idx].generated_code,
                }
            )
            ctr += 1

            if ctr == 10:
                break

        all_results.append(results)

    return all_results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    fire.Fire(
        {
            "prepare_dataset_for_generational_model": prepare_dataset_for_generational_model,
            "train_generational_model": train_generational_model,
            "prepare_dataset_for_training_embeddings": prepare_dataset_for_training_embeddings,
            "train_embeddings": train_embeddings,
            "prepare_search_engine": prepare_search_engine,
            "start_search_engine": start_search_engine,
        }
    )
