
cuda=2

for model in 01; do
for alpha in 15.0; do

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
generate.py \
    --collator_name gpt2_eval \
    --model_name gpt2 \
    --pretrained_model_path checkpoints_senti/neg_contrast2_${model}/${alpha} \
    --save_path checkpoints_senti/neg_contrast2_${model}/${alpha} \
    --infer_data_paths data_senti/gpt2/neutral.txt data_senti/gpt2/positive.txt \
    --infer_names neutral positive \
    --only_generate \
    --max_input_length 15 \
    --max_decoder_input_length 20 \
    --seed 0 \
    --max_length 20 \
    --min_length 15 \
    --batch_size 16 \
    --temperature 1 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --num_return_sequences 25 \
    --length_penalty 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 0 \
    --encoder_no_repeat_ngram_size 0

done
done
