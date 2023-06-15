
cuda=7
p=0.7
num=4

CUDA_VISIBLE_DEVICES=${cuda} accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
generate.py \
    --collator_name gpt2_eval \
    --model_name gpt2 \
    --pretrained_model_path checkpoints_wiki/gpt2 \
    --save_path checkpoints_wiki/gpt2/augment/train_${p} \
    --infer_data_paths data_wiki/gpt2/train_augment.txt \
    --infer_names train \
    --only_generate \
    --max_input_length 32 \
    --max_decoder_input_length 128 \
    --seed 0 \
    --max_length 128 \
    --min_length 64 \
    --batch_size 128 \
    --temperature 1 \
    --top_k 0 \
    --top_p ${p} \
    --num_beams 1 \
    --num_return_sequences ${p} \
    --length_penalty 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 0 \
    --encoder_no_repeat_ngram_size 0
