import os
from os import listdir
from os.path import join
import torch
from torch import autocast
from pathlib import Path
import random
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse
from utils import load_list
from tqdm import tqdm
import math
from glob import glob 

from load_data import get_class_name, get_class_index, get_id_class_name_map_dict


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imagenet1k", help="dataset")
parser.add_argument("--n_img_per_class", type=int, default=1300, help="number of generated images for each class")
parser.add_argument("--guidance_scale", type=float, default=2, help="guidance scale")
parser.add_argument("--data_dir", type=str, default="/media/slei/slei_disk/data/curated_imagenet", help="path to save generated dataset")
parser.add_argument('--use_caption', action='store_true')
parser.add_argument("--caption_dir", type=str, default="/media/slei/slei_disk/distill_image_to_text/imagenet_caption_blip2", help="path of image captioning")
parser.add_argument("--n_gpus_for_one_dataset", type=int, default=500, help="xxx")
parser.add_argument("--data_piece", type=int, default=0, help="xxx")
args = parser.parse_args()

# Print Args
print("--------args----------")
for k in list(vars(args).keys()):
    print("%s: %s" % (k, vars(args)[k]))
print("--------args----------\n")


dataset = args.dataset
n_img_per_class = args.n_img_per_class
guidance_scale = args.guidance_scale
data_dir = args.data_dir
use_caption = args.use_caption
caption_dir = args.caption_dir
n_gpus_for_one_dataset = args.n_gpus_for_one_dataset
data_piece = args.data_piece

if use_caption:
    subset_name = dataset + "_use_caption_gs_" + str(guidance_scale)
else:
    subset_name = dataset + "_gs_" + str(guidance_scale)

out_dir = Path(os.path.join(data_dir + f'_{n_img_per_class}', subset_name))
out_dir.mkdir(exist_ok=True, parents=True)

dict_id_to_class_name, dict_class_name_to_id = get_id_class_name_map_dict()

# divide the classes
class_names = get_class_name(dataset)

n_divide = len(class_names) // n_gpus_for_one_dataset
class_names = class_names[data_piece * n_divide: (data_piece + 1) * n_divide]

if use_caption:
    class_index = get_class_index(dataset)
    class_index = class_index[data_piece * n_divide: (data_piece + 1) * n_divide]

# caption_name_list = listdir(caption_dir)
caption_name_list = [dict_class_name_to_id[i] for i in class_names]
caption_path_list = [join(caption_dir, f) for f in caption_name_list]

# Import stable diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#pipe.scheduler =  DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
num_inference_steps = 50
pipe = pipe.to(device)
batch_size = 2

if use_caption:
    for c_index, c in tqdm(enumerate(class_names), total=len(class_names)):
        print("Generate %s-th class."%{c_index})
        caption_path = caption_path_list[c_index]
        caption = load_list(caption_path)
        
        sub_dir = Path(os.path.join(out_dir, c))
        sub_dir.mkdir(exist_ok=True, parents=True)
        
        # check existing files
        file_list = sorted(glob(os.path.join(sub_dir, '*jpg')))
        total_img_cnt = len(file_list)
        
        while total_img_cnt <= n_img_per_class:
            for i in range(0, len(caption), batch_size):
                prompt_list = []
                for j in range(batch_size):
                    if i + j >= len(caption):
                        break
                    prompt = f'a photo of {c}, ' + caption[i + j]
                    prompt_list.append(prompt)
                if len(prompt_list) == 0:
                    continue
                seed = random.randint(0, 130000)
                with autocast('cuda'):
                    generator = torch.Generator(device=device).manual_seed(seed)
                    images, nsfws = pipe(prompt_list, generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, return_dict=False, num_images_per_prompt=1)

                    for img, nsfw in zip(images, nsfws):
                        if nsfw:
                            continue
                        
                        total_img_cnt += 1
                        outpath = sub_dir /  f"{c}_{total_img_cnt}.jpg"
                        img.save(outpath)
    else:
        for c in class_names:

            prompt = f'a photo of {c}'
            
            sub_dir = Path(os.path.join(out_dir, c))
            sub_dir.mkdir(exist_ok=True, parents=True)


            # TODO: rewrite this part

            for i in range(n_img_per_class):
                seed = random.randint(0, 10000)
                with autocast('cuda'):
                    generator = torch.Generator(device=device).manual_seed(seed)
                    nsfw = True
                    while nsfw:
                        out = pipe(prompt, generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
                        nsfw = out["nsfw_content_detected"][0] # avoid saving NSFW/black images
                
                image = out.images[0]
                outpath = sub_dir /  f"{c}_{i}.jpg"
                image.save(outpath)
