
## Reproduce the paper results

### Download the dataset

```bash
huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train --local-dir $WORK_DIR/data/nq_hotpotqa_train
```

### Launch the local search engine

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

### Run PPO training
```bash
bash train_ppo.sh
```


### Run GRPO training
```bash
bash train_grpo.sh
```

### Run evaluation
```bash
bash evaluate.sh
```

You can change ```$BASE_MODEL``` to the path of the model you would like to evaluate.
