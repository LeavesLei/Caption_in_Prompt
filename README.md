# Caption in Prompt

Official repository for the IJCV paper "[Image Captions are Natural Prompts for Text-to-Image Models (arXiv version)](https://arxiv.org/abs/2307.08526)" by Shiye Lei\*, Hao Chen\*, Sen Zhang, Bo Zhao, and Dacheng Tao.

\* Contributed equally.

(The repository is still under construction)

## Dependencies

- Python 3.7
- Pytorch 1.11
- transformers
- diffusers

## Quick Start

#### Image Caption Generation
`python image_capotion_generate.py`

#### Generate Synthetic ImageNet1K
- **With caption**

`python image_dataset_generate.py --use_caption`
- **With basic prompt**

`python image_dataset_generate.py`

#### Model Training on Syn Data
`python train_timm.py --curated_dataset [SYN DATA PATH]`

#### Evaluation

- **In and Out Distribution Accuracy Evaluation**
  
`python eval_model.py`

- **Membership Inference Attack**

`eval_mi_attack.py`
## Contact

For any issue, please kindly contact Shiye Lei: [leishiye@gmail.com](mailto:leishiye@gmail.com)
