
cuda=6
seed=0

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
generate.py \
    --collator_name gpt2_eval \
    --model_name gpt2 \
    --pretrained_model_path checkpoints_senti/gpt2_both \
    --save_path checkpoints_senti/gpt2_both \
    --infer_data_paths data_senti/gpt2/augment.txt \
    --infer_names augment \
    --only_generate \
    --max_input_length 2 \
    --max_decoder_input_length 33 \
    --seed ${seed} \
    --max_length 33 \
    --min_length 20 \
    --batch_size 16 \
    --temperature 1 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --num_return_sequences 20 \
    --length_penalty 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 0 \
    --encoder_no_repeat_ngram_size 0
