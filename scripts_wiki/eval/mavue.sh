
paths=()
paths[${#paths[*]}]="checkpoints_wiki/contrast_03/15.0/test_greedy"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_mauve.py \
    --context_file data_wiki/gpt2/test.txt \
    --infer_data_paths ${paths[*]} \
    --batch_size 320
