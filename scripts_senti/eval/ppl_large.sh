
for domain in positive negative neutral; do

paths=()
for setting in pos neg; do
paths[${#paths[*]}]="checkpoints_senti/${setting}_contrast_01/15.0/${domain}"
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
eval_ppl_gpt2.py \
    --save_name large \
    --pretrained_model_path /home/zhengchujie/pretrained-models-large/gpt2-xl \
    --context_file data_senti/gpt2/${domain}.txt \
    --infer_data_paths ${paths[*]} \
    --max_input_length 15 \
    --max_decoder_input_length 20 \
    --seed 42 \
    --batch_size 1600

done
