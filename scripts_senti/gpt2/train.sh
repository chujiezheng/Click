
CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
train.py \
    --collator_name gpt2 \
    --model_name gpt2 \
    --pretrained_model_path /home/zhengchujie/pretrained-models-large/gpt2-large \
    --save_path checkpoints_senti/gpt2_both \
    --train_data_path data_senti/gpt2/train_both.txt \
    --max_input_length 35 \
    --seed 42 \
    --adafactor \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 50

bash scripts_senti/gpt2/augment_both.sh
