# Caption in Prompt

Official repository for the IJCV paper "**Image Captions are Natural Prompts for Training Data Synthesis** [[Springer]](https://link.springer.com/article/10.1007/s11263-025-02436-0) [[arXiv]](https://arxiv.org/pdf/2307.08526)" by Shiye Lei\*, Hao Chen\*, Sen Zhang, Bo Zhao, and Dacheng Tao. 

\* Contributed equally.

## Dependencies

- Python 3.7
- Pytorch 1.11
- transformers
- diffusers

## Quick Start

- We provide the image captions extracted by [ViT-GPT2](https://github.com/LeavesLei/Caption_in_Prompt/blob/main/imagenet_caption_vit-gpt2.zip) and [BLIP2](https://github.com/LeavesLei/Caption_in_Prompt/blob/main/imagenet_caption_blip2.zip).
- The ResNet-50 models trained on synthetic ImageNet are [available](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/slei5230_uni_sydney_edu_au/EQxATumNWb9GhcbXp9XdDiQBjQSEgnuZ2tWRsk0R-yXPbw?e=PhO0Ff).
  
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

#### Citation
```
@article{lei2025image,
  title={Image Captions are Natural Prompts for Training Data Synthesis},
  author={Lei, Shiye and Chen, Hao and Zhang, Sen and Zhao, Bo and Tao, Dacheng},
  journal={International Journal of Computer Vision},
  pages={1--20},
  year={2025},
  publisher={Springer}
}
```
## Contact

For inquiries regarding synthetic data requirements or any other issues, please feel free to contact Shiye Lei: [leishiye@gmail.com](mailto:leishiye@gmail.com)
