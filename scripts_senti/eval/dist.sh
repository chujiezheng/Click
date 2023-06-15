
for domain in neutral positive negative; do

paths=()
for setting in pos neg; do
paths[${#paths[*]}]="checkpoints_senti/${setting}_contrast_01/15.0/${domain}"
done

python eval_dist.py \
    --context_file data_senti/gpt2/${domain}.txt \
    --infer_data_paths ${paths[*]} \

done
