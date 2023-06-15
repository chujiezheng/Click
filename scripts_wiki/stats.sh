
paths=()
paths[${#paths[*]}]="checkpoints_wiki/contrast_03/15.0/test_greedy"

python results_wiki.py --infer_data_paths ${paths[*]} --test
