
paths=()
paths[${#paths[*]}]="checkpoints_bad/ft/test"
paths[${#paths[*]}]="checkpoints_bad/unlikelihood_01/test"
paths[${#paths[*]}]="checkpoints_bad/gedi/test_20.0"
paths[${#paths[*]}]="checkpoints_bad/dexperts/test_5.0"
paths[${#paths[*]}]="checkpoints_bad/director_02/test_10.0"
paths[${#paths[*]}]="checkpoints_bad/cringe_02/test"
paths[${#paths[*]}]="checkpoints_bad/contrast_05/20.0/test"

python eval_dist.py \
    --context_file data_bad/blender/test.txt \
    --infer_data_paths ${paths[*]} \

