
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
eval_senti.py \
    --collator_name classification \
    --model_name distilbert \
    --pretrained_model_path /home/zhengchujie/pretrained-models/distilbert-base-uncased-finetuned-sst-2-english \
    --model_args 2 \
    --context_file data_senti/gpt2/augment.txt \
    --infer_data_paths checkpoints_senti/gpt2_both/augment \
    --max_input_length 45 \
    --seed 42 \
    --batch_size 800
