import os
import json
import random
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from .utils import load_jsonl, lower_keys
import pdb

from datasets import Dataset, concatenate_datasets
import pandas as pd


def stratified_sample(
    dataset: Dataset, N: int, group_col: str = "source", seed: int = 42
) -> Dataset:
    """
    Perform stratified sampling of N examples from a HuggingFace dataset.
    Proportions are preserved according to counts in group_col.
    Uses Hamilton's (largest remainder) method to allocate samples.
    """
    df = dataset.to_pandas()
    counts = df[group_col].value_counts()

    # Proportional allocation
    weights = counts / counts.sum()
    targets = weights * N
    floors = targets.astype(int)
    remainders = targets - floors
    remaining = N - floors.sum()
    allocation = floors.copy()
    allocation.loc[remainders.sort_values(ascending=False).index[:remaining]] += 1

    # Draw samples per group
    sampled_subsets = []
    for src, n in allocation.items():
        subset = dataset.filter(lambda x, s=src: x[group_col] == s)
        sampled_subsets.append(subset.shuffle(seed=seed).select(range(n)))

    # Concatenate all sampled subsets
    return concatenate_datasets(sampled_subsets)


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "svamp":
            # evaluate on training set + test set
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets(
                [dataset, load_dataset("ChilleD/SVAMP", split="test")]
            )
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(
                lambda x: ";" not in x["answer"]
            )  # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example["type"] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", "all", split="test")
            # only keep stem subjects
            stem_subjects = [
                "abstract_algebra",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
                "machine_learning",
            ]
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x["type"] in stem_subjects)
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        elif data_name == "prm800k":
            dataset = load_dataset("HuggingFaceH4/prm800k-trl-dedup", split="train")
            dataset = dataset.to_list()
            # remove duplicates
            dataset = remove_prompt_duplicates(dataset, key="prompt")
        elif data_name == "open-r1":
            dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
            dataset = dataset.to_list()
            dataset = remove_prompt_duplicates(dataset, key="problem")
        elif data_name == "NuminaMath-CoT":
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
            dataset = remove_prompt_duplicates(dataset, key="problem")
            # stratified sampling
            dataset = stratified_sample(dataset, N=30000, group_col="source")
            dataset = dataset.to_list()
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def remove_prompt_duplicates(dataset, key="prompt"):
    """
    Remove duplicates from a dataset based on the 'prompt' key.

    Args:
        dataset (Dataset): Dataset to remove duplicates from
        key (str): Key to remove duplicates on

    Returns:
        Dataset: Dataset with duplicates removed
    """
    df = dataset.to_pandas()
    df = df.drop_duplicates(subset=[key])
    dataset = Dataset.from_pandas(df)
    return dataset


def remove_prompt_duplicates_previous(dict_list, key="prompt"):
    """
    Remove duplicates from a list of dictionaries based on the 'prompt' key.
    Keeps the first occurrence of each prompt.

    Args:
        dict_list (list): List of dictionaries, where each dictionary contains a 'prompt' key

    Returns:
        list: New list with duplicates removed
    """
    seen_prompts = {}
    unique_list = []

    for item in dict_list:
        # Skip if the item doesn't have a 'prompt' key
        if key not in item:
            continue

        prompt = item[key]
        if prompt not in seen_prompts:
            seen_prompts[prompt] = True
            unique_list.append(item)

    return unique_list
