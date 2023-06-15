CUDA_VISIBLE_DEVICES=7 accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
train.py \
    --collator_name classification \
    --model_name roberta \
    --pretrained_model_path /home/zhengchujie/pretrained-models/roberta-base \
    --model_args 2 \
    --save_path checkpoints_cls/bad \
    --train_data_path data_cls/bad/train.txt \
    --max_input_length 192 \
    --seed 42 \
    --adafactor \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --warmup_steps 50
