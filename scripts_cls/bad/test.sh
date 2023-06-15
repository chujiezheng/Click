CUDA_VISIBLE_DEVICES=7 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
infer.py \
    --collator_name classification \
    --model_name roberta \
    --pretrained_model_path checkpoints_cls/bad \
    --model_args 2 \
    --save_path checkpoints_cls/bad \
    --max_input_length 192 \
    --seed 0 \
    --batch_size 256 \
    --infer_data_paths data_cls/bad/valid.txt data_cls/bad/test.txt \
    --infer_names valid test
