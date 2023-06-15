
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 32 \
eval_ppl_blender.py \
    --save_name self \
    --pretrained_model_path checkpoints_bad/contrast_05/20.0 \
    --context_file data_bad/blender/train.txt \
    --infer_data_paths checkpoints_bad/contrast_05/20.0/train \
    --max_input_length 128 \
    --max_decoder_input_length 32 \
    --seed 42 \
    --batch_size 1600

