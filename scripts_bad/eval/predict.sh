
paths=()
paths[${#paths[*]}]="checkpoints_bad/ft/test"
paths[${#paths[*]}]="checkpoints_bad/unlikelihood_01/test"
paths[${#paths[*]}]="checkpoints_bad/gedi/test_20.0"
paths[${#paths[*]}]="checkpoints_bad/dexperts/test_5.0"
paths[${#paths[*]}]="checkpoints_bad/director_02/test_10.0"
paths[${#paths[*]}]="checkpoints_bad/cringe_02/test"
paths[${#paths[*]}]="checkpoints_bad/contrast_05/20.0/test"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
eval_bad.py \
    --collator_name classification \
    --model_name roberta \
    --pretrained_model_path checkpoints_cls/bad \
    --model_args 2 \
    --context_file data_bad/raw/test.txt \
    --infer_data_paths ${paths[*]} \
    --max_input_length 192 \
    --seed 42 \
    --batch_size 800
