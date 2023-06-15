
setting=pos
paths=()
paths[${#paths[*]}]="checkpoints_senti/${setting}_contrast_01/15.0"

python results_senti.py --infer_data_paths ${paths[*]} --positive
