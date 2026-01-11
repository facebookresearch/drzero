# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs
from search.index_builder import load_corpus
from verl.prompts import *


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HOPS = [1, 2, 3, 4]
HOP_RATIO = [4, 3, 2, 1]
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


def process_single_row(row, corpus_iter, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    # question = row.get("question", "")  # not used

    # Build prompt structure
    doc = next(corpus_iter)['contents']
    doc_title = doc.split("\n")[0]
    doc_text = doc.split("\n")[1]

    raw_document = f"(Title: {doc_title})\n{doc_text}\n"
    encoded_document = TOKENIZER.encode(raw_document)[:256]
    decoded_document = TOKENIZER.decode(encoded_document)

    n_hop = np.random.choice(HOPS, size=1, p=np.array(HOP_RATIO)/sum(HOP_RATIO))[0]    
    user_content = user_content_prefix.format(hops=n_hop, document=decoded_document)
    prompt = [{"role": "user", "content": user_content}]

    # Process data source
    data_source_tagged = f"search_zero_{n_hop}"
    reward_model = row.get("reward_model")
    reward_model['ground_truth']['target'] = None

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": "", "question": "", "data_source": data_source_tagged
            }
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []
    corpus_iter = iter(load_corpus(args.corpus_dir).shuffle(seed=42))

    # Download and process files using temporary directory
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train"]:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            # Download Parquet file from HuggingFace
            logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
            local_parquet_filepath = hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=parquet_filename,
                repo_type="dataset",
                local_dir=tmp_download_dir,
                local_dir_use_symlinks=False,
            )

            # Load and process Parquet file
            df_raw = pd.read_parquet(local_parquet_filepath)
            logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

            def apply_process_row(row, split_name=split):
                return process_single_row(row, corpus_iter, current_split_name=split_name, row_index=row.name)

            df_processed = df_raw.apply(apply_process_row, axis=1)

            # Save processed DataFrame
            ratio_postfix = "ratio" + "".join(str(x) for x in HOP_RATIO)
            output_file_path = os.path.join(local_save_dir, f"zero_{ratio_postfix}.parquet")
            df_processed.to_parquet(output_file_path, index=False)
            logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
            processed_files.append(output_file_path)

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    print("Example prompt: ", df_processed.prompt[0])
    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Search-R1 from HuggingFace, process, and save to Parquet."
    )
    parser.add_argument(
        "--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir",
        default="./data",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")
    parser.add_argument("--corpus_dir", default="./corpus/wiki-18.jsonl", help="Path to Wiki corpus data.")
    args = parser.parse_args()

    user_content_prefix = DEFAULT_CHALLENGER_PREFIX

    main()