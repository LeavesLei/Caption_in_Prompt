# Caption in Prompt

Official repository for the IJCV 2025 paper "[Image Captions are Natural Prompts for Text-to-Image Models (arXiv version)](https://arxiv.org/abs/2307.08526)" by Shiye Lei\*, Hao Chen\*, Sen Zhang, Bo Zhao, and Dacheng Tao.

\* Contributed equally.

(The repository is still under construction)

## Dependencies

- Python 3.7
- Pytorch 1.11


Download ImageNet_Caption_Blip2: https://drive.google.com/file/d/1wS-ikxydoZeHXjQspJTIT0rNiq-t-zF9/view?usp=sharing

`image_dataset_generate.py`:
- `guidance_scale` = [1, 1.5, 2, 2.5. 3, 4, 5, 6, 7.5]
- `data_dir`:  path to save the generated dataset
- `use_caption`
- `caption_dir`: path of image caption file

`eval_dataset.py`:
- `guidance_scale`
- `data_path`: same as the data_dir
- `ImageNetPath`: path of ImageNet
- `use_caption`


## Contact

For any issue, please kindly contact Shiye Lei: [leishiye@gmail.com](mailto:leishiye@gmail.com)
