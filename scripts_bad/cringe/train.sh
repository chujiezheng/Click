
cuda=0

for gamma in 02; do

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
train.py \
    --collator_name text2text_labels \
    --model_name blender_cringe_${gamma} \
    --pretrained_model_path /home/zhengchujie/pretrained-models/facebook/blenderbot-400M-distill \
    --save_path checkpoints_bad/cringe_${gamma} \
    --train_data_path data_bad/labels/train.txt \
    --max_input_length 128 \
    --max_decoder_input_length 32 \
    --seed 42 \
    --adafactor \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 50

done
