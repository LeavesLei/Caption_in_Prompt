# caption_to_image

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
