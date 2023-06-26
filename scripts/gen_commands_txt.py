
import os
import numpy as np

def write_diffusion_jobs(
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
    caption_dir = f'/zhaobai46e/haoc/caption_to_image-amlt/caption_data/{caption_data}_caption_{caption_model}'
    base_data_dir = f'/zhaobai46e/haoc/caption_to_image-amlt/diffusion_data/{dataset}'

    # data_piece_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    totalgpus = 50
    data_piece_list = np.arange(50)
    jobs_list = []
    job_cnt = 0
    for guidance_scale in guidance_list:
        for use_caption in caption_list:
            for data_piece in data_piece_list:
                data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                job_name = f'{dataset}_scale{guidance_scale}_caption{caption_model}_piece{data_piece}'

                if use_caption:
                    command = f'python image_dataset_generate_imagenet1k.py --dataset {dataset} --guidance_scale {guidance_scale} --use_caption --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece} --n_gpus_for_one_dataset {totalgpus} > ./logs_synthesize/{job_name}.log 2>&1'
                else:
                    command = f'python image_dataset_generate_imagenet1k.py --dataset {dataset} --guidance_scale {guidance_scale} --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece} --n_gpus_for_one_dataset {totalgpus} > ./logs_synthesize/{job_name}.log 2>&1'

                jobs_list.append(command)
                job_cnt += 1

    jobs_list.sort()
    num_nodes = 1
    step = 8 # job_cnt // num_nodes
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



def write_llm_jobs(
    dataset='imagenet1k',
    llm_model='vicuna-13b-v1.3',
    caption_data='imagenet',
    caption_model='blip2'):

    n_gpus_for_one_dataset = 50
    data_piece_list = np.arange(n_gpus_for_one_dataset)
    
    controller_ports = range(10001, 11120, 1)
    worker_ports = range(20120, 21120, 1)
    api_ports = range(30120, 31120, 1)
    
    caption_dir = f'/zhaobai46e/haoc/caption_to_image-amlt/caption_data/{caption_data}_caption_{caption_model}'
    base_save_dir = f'/zhaobai46e/haoc/caption_to_image-amlt/caption_data/llm_rewrite/20230624/{caption_data}_caption_{caption_model}_llm_rewrite_raw'
    os.makedirs(base_save_dir, exist_ok=True)
    
    jobs_list = []
    for i, data_piece in enumerate(data_piece_list):
        job_name = f'llm_rewrite_{caption_data}_{caption_model}_{data_piece}'
        command_0 = f'nohup python3 -m fastchat.serve.controller --host 127.0.0.1 --port {controller_ports[i]} > controller.log &' 
        command_1 = f'nohup python3 -m fastchat.serve.model_worker --host 127.0.0.1 --port {worker_ports[i]} --worker-address http://127.0.0.1:{worker_ports[i]} --controller-address --worker-address http://127.0.0.1:{controller_ports[i]}  --model-path lmsys/{llm_model} > model_worker.log &'
        command_2 = f'sleep 60'
        command_3 = f'nohup python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port {api_ports[i]} --controller-address http://127.0.0.1:{controller_ports[i]}  > api_server.log &'
        command_4 = f'sleep 60'
        command_5 = f'python3 llm_rewrite.py --caption-dir {caption_dir} --save-dir {base_save_dir} --llm-per-caption 5 --llm-tempreture 0.5 --port {api_ports[i]} --model {llm_model} --n_gpus_for_one_dataset {n_gpus_for_one_dataset} --data_piece {data_piece}'
        jobs_list.append((job_name, [command_0, command_1, command_2, command_3, command_4, command_5]))

    os.makedirs('/zhaobai46e/fastchat/command_config/', exist_ok=True)
    os.makedirs('/zhaobai46e/fastchat/log_llm/', exist_ok=True)
    base_name = f'/zhaobai46e/fastchat/command_config/{dataset}_{caption_model}_llm'

    for i, job in enumerate(jobs_list):
        
        job_name, command_list = job 
        
        with open(base_name + '_' + str(i) + '.sh', 'w') as f:
            
            f.write('exec bash\n')
            f.write('conda activate test')
            for command in command_list:
                f.write(command + '\n')
                
    print(f"Generate {len(jobs_list)} tasks in {len(jobs_list)} files")



if __name__ == '__main__':
    write_diffusion_jobs(dataset='imagenet1k', caption_model='blip2')
    write_llm_jobs(caption_model='syn_blip2')