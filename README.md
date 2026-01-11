# Self-Evolving Search Agents without Training Data

This repository contains the code for **Dr. Zero: Self-Evolving Search Agents without Training Data**, a framework enabling search agents to evolve and improve their reasoning and retrieval capabilities without relying on any training data.

## 🚀 Overview

The core idea is to bootstrap a search agent from a base model (e.g., Qwen or Llama) through multiple iterations of data-free self-play and reinforcement learning in a multi-turn tool-using environment.

*   **Solver:** The primary search agent learning to answer queries using the search tool.
*   **Challenger:** An adversarial or auxiliary agent that evolves to creates harder questions and thereby driving the curriculum.
*   **Zero-Shot Initialization:** The process starts with zero training data and an external search engine (e.g., Wikipedia passage retriever).

## 🛠️ Setup & Installation

### 1. Environment

Ensure you have a Python environment with the necessary dependencies (PyTorch, transformers, faiss, verl==0.5.0, etc.). The rest of the dependencies can be found [here](https://github.com/volcengine/verl/blob/v0.5.0/requirements.txt) and [here](https://github.com/volcengine/verl/blob/v0.5.0/requirements_sglang.txt).

### 2. Search Engine

This framework relies on a local server with an embedding model. You need to prepare the corpus and build the index before running any training.

**Download Corpus / Index:**
Use the provided script to build a FAISS index for the retriever (default: `intfloat/e5-base-v2`).

```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

## 🔄 Iterative Self-Evolution Workflow

The training process proceeds in iterations (Iter 1, Iter 2, Iter 3...). Each iteration typically consists of three phases:

### Phase 0: Initial Data Preparation

Before the first iteration, prepare the initial synthetic dataset (prompt only, not actual questions / answers) and the evaluation data.

```bash
python process_train.py --local_dir ./data
python process_test.py --local_dir ./data
```

### Iteration 1

**1. Train Challenger (Proposer):**
Train the Challenger agent. The proposer agent is incentivized to generate challenging yet manageable questions for the solver.

```bash
bash iter1_challenger.sh
```

**2. Generate Data:**
Generate training data using the learnt proposer.

```bash
bash iter1_gen_data.sh
```

**3. Train Solver:**
Train the solver agent on the generated data. This optimizes the agent's ability to search and reason.

```bash
bash iter1_solver.sh
```

### Subsequent Iterations (Iter 2, Iter 3...)

Repeat the process using the scripts for the respective iteration. The model checkpoints from the previous iteration are used as the starting point for the next.

*   `iter2_challenger.sh` -> `iter2_gen_data.sh` -> `iter2_solver.sh`
*   `iter3_challenger.sh` -> `iter3_gen_data.sh` -> `iter3_solver.sh`

## Acknowledgements

*   [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
*   [VeRL](https://github.com/volcengine/verl)
