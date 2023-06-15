
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
    --mixed_precision fp16 \
    --num_processes 4 \
    --multi_gpu \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
    --main_process_port 29504 \
train.py \
    --collator_name gpt2 \
    --model_name gpt2 \
    --pretrained_model_path /home/zhengchujie/pretrained-models/gpt2-small \
    --save_path checkpoints_wiki/gpt2 \
    --train_data_path data_wiki/gpt2/train.txt \
    --max_input_length 256 \
    --seed 42 \
    --adafactor \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --num_optim_steps 40000 \
    --warmup_steps 4000
