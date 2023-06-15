
cuda=3

for model in 03; do
for alpha in 15.0; do

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
generate.py \
    --collator_name gpt2_eval \
    --model_name gpt2 \
    --pretrained_model_path checkpoints_wiki/contrast_${model}/${alpha} \
    --save_path checkpoints_wiki/contrast_${model}/${alpha} \
    --infer_data_paths data_wiki/gpt2/valid.txt data_wiki/gpt2/test.txt \
    --infer_names valid_greedy test_greedy \
    --max_input_length 35 \
    --max_decoder_input_length 128 \
    --seed 0 \
    --max_length 128 \
    --min_length 64 \
    --batch_size 64 \
    --temperature 1 \
    --top_k 0 \
    --top_p 1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --length_penalty 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 0 \
    --encoder_no_repeat_ngram_size 0

done
done
