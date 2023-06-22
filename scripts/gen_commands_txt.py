
import os
import numpy as np

def write_diffusion_jobs_imagenet1k(
               sku='G1', 
               dataset='imagenet1k',
               caption_data='imagenet',
               caption_model='blip2'):

    # guidance_list = [1.5, 2]
    guidance_list = [1.5]
    if caption_model is not None:
        caption_list = [True]
    else:
        caption_list = [False]
    caption_dir = f'/zhaobai46e/haoc/caption_to_image/caption_data/{caption_data}_caption_{caption_model}'
    base_data_dir = f'/zhaobai46e/haoc/caption_to_image/caption_data/{dataset}'

    # data_piece_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data_piece_list = np.arange(32)
    jobs_list = []
    job_cnt = 0
    for guidance_scale in guidance_list:
        for use_caption in caption_list:
            for data_piece in data_piece_list:
                data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                job_name = f'{dataset}_scale{guidance_scale}_caption{caption_model}_piece{data_piece}'
                
                if use_caption:
                    command = f'python image_dataset_generate_imagenet1k.py --dataset {dataset} --guidance_scale {guidance_scale} --use_caption --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece} > ./logs_synthesize/{job_name}.log 2>&1'
                else:
                    command = f'python image_dataset_generate_imagenet1k.py --dataset {dataset} --guidance_scale {guidance_scale} --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece} > ./logs_synthesize/{job_name}.log 2>&1'

                jobs_list.append(command)
                job_cnt += 1
    
    jobs_list.sort()
    num_nodes = 4
    step = job_cnt // num_nodes
    os.makedirs('command_config', exist_ok=True)
    os.makedirs('logs_synthesize', exist_ok=True)
    base_name = f'command_config/{dataset}_{caption_model}_synthesize'
    node_cnt = 0
    for i in range(0, job_cnt, step):
        jobs = jobs_list[i:i+step]
        with open(base_name + '_' + str(node_cnt) + '.txt', 'w') as f:
            for job in jobs:
                f.write(job + '\n')
        node_cnt += 1
    print(f"Generate {len(jobs_list)} tasks in {node_cnt} files")


if __name__ == '__main__':
    write_diffusion_jobs_imagenet1k(dataset='imagenet1k', caption_model='blip2')