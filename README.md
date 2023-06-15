# Click
This is the official repository for the Findings of ACL 2023 paper "[Click: Controllable Text Generation with Sequence Likelihood Contrastive Learning](https://arxiv.org/abs/2306.03350)".

## Data, Models, and Generation Results

You can download our used data and trained models in [HuggingFace](https://huggingface.co/chujiezheng/click) (these files may be a bit large).

To reproduce the experiments, you need to download the `data_*` folders and put them in the project path.

## Requirements

**Note**: The listed packages may be not inclusive. You may install additional packages according to the error information.

```conda
torch
transformers
accelerate
```

## Detoxification Experiments

We additionally release the implementation of Click as well as the baselines (see `models`). The running scripts are in `scripts_bad`. We take Click for instance to illustrate how to run the codes.

**NOTE**: `scripts_cls`, `data_cls`, and `checkpoints_cls` contains scripts, data, and checkpoints of our trained BAD classifier for evaluation.

```bash
# process raw data
# you may need to download the original datasets and change the data path
cd /{project_path}/data_bad/raw
python process.py

# process train/valid/test data
cd /{project_path}/data_bad/blender
python process.py

# generate multiple continuations
cd /{project_path}
bash scripts_bad/blender/augment.sh
bash scripts_bad/eval/ppl_self.sh
bash scripts_bad/eval/predict_train.sh

# process contrastive learning data
cd /{project_path}/data_bad/contrast
python process.py

# run training and generation
# you may use scripts for different models
cd /{project_path}
bash scripts_bad/contrast/train.sh
bash scripts_bad/contrast/generate.sh
# ... and you may generate multiple continuations for the next iteration
# bash scripst_bad/contrast/augment.sh

# run evaluation
# you may change the arguments as needed
cd /{project_path}
bash scripts_bad/eval/dist.sh
bash scripts_bad/eval/ppl_large.sh
bash scripts_bad/eval/predict.sh

# gather results
cd /{project_path}
bash scripts_bad/stats.sh

# then look into `results_bad_test.txt` for results
# the results' order is consistent as presented in the paper
# you can paste them into Excel :)
```

## Sentiment Experiments

```bash
# process raw data
# you may need to change the data path
cd /{project_path}/data_senti/gpt2
python process.py

# generate multiple continuations
cd /{project_path}
bash scripts_senti/gpt2/train.sh
bash scripts_senti/gpt2/augment.sh
bash scripts_senti/eval/ppl_self.sh
bash scripts_senti/eval/predict_train.sh

# process contrastive learning data
cd /{project_path}/data_senti/contrast
python process.py

# you may use pos/neg to specify the domain
# run training and generation
cd /{project_path}
bash scripts_senti/pos_contrast/train.sh
bash scripts_senti/pos_contrast/generate.sh

# run evaluation
# you may change the arguments as needed
cd /{project_path}
bash scripts_senti/eval/dist.sh
bash scripts_senti/eval/ppl_large.sh
bash scripts_senti/eval/predict.sh

# gather results
cd /{project_path}
bash scripts_senti/stats_pos.sh

# then look into `results_senti_pos.txt` for results
```

## Repetition Experiments

**NOTE**: Since the training files for Click are too large (wikitext-103), I did not upload them on HuggingFace. But you may reproduce the data by running the following codes (the trained models used for producing the Click training data are uploaded).

```bash
# process raw data
# you may need to change the data path
cd /{project_path}/data_wiki/gpt2
python process.py

# generate multiple continuations
cd /{project_path}
bash scripts_wiki/gpt2/train.sh
# note: you should reproduce the Click training data from here
bash scripts_wiki/gpt2/augment_0.5.sh
bash scripts_wiki/gpt2/augment_0.7.sh
bash scripts_wiki/gpt2/augment_0.9.sh
bash scripts_wiki/eval/div.sh

# process contrastive learning data
cd /{project_path}/data_wiki/contrast
python process.py

# you may use pos/neg to specify the domain
# run training and generation
cd /{project_path}
bash scripts_wiki/contrast/train.sh
bash scripts_wiki/contrast/generate.sh

# run evaluation
# you may change the arguments as needed
cd /{project_path}
bash scripts_wiki/eval/mauve.sh

# gather results
cd /{project_path}
bash scripts_wiki/stats.sh

# then look into `results_wiki_test.txt` for results
```

## Citation

Please kindly cite our paper if our paper and codes are helpful.

```bib
@inproceedings{zheng-etal-2023-click,
  title={Click: Controllable Text Generation with Sequence Likelihood Contrastive Learning},
  author={Zheng, Chujie and
    Ke, Pei and
    Zhang, Zheng and
    Huang, Minlie},
  booktitle={Findings of ACL},
  year={2023}
}
```
