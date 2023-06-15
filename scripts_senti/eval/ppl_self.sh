
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
eval_ppl_gpt2.py \
    --save_name self \
    --pretrained_model_path checkpoints_senti/gpt2_both \
    --context_file data_senti/gpt2/augment.txt \
    --infer_data_paths checkpoints_senti/gpt2_both/augment \
    --max_input_length 2 \
    --max_decoder_input_length 33 \
    --seed 42 \
    --batch_size 1600
