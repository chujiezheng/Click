
cuda=0

for model in 03; do
for alpha in 15.0; do

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --mixed_precision fp16 \
    --num_processes 4 \
    --multi_gpu \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
    --main_process_port 29504 \
train.py \
    --collator_name gpt2_contrast \
    --model_name gpt2_contrast_${model} \
    --pretrained_model_path checkpoints_wiki/gpt2 \
    --model_args ${alpha} \
    --save_path checkpoints_wiki/contrast_${model}/${alpha} \
    --train_data_path data_wiki/contrast/train.txt \
    --max_input_length 256 \
    --max_decoder_input_length 160 \
    --seed 42 \
    --adafactor \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_optim_steps 5000 \
    --warmup_steps 200

done
done
