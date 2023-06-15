
paths=()
paths[${#paths[*]}]="checkpoints_wiki/gpt2/augment/train_0.5"
paths[${#paths[*]}]="checkpoints_wiki/gpt2/augment/train_0.7"
paths[${#paths[*]}]="checkpoints_wiki/gpt2/augment/train_0.9"

python eval_div.py \
    --context_file data_wiki/gpt2/train_augment.txt \
    --infer_data_paths ${paths[*]}
