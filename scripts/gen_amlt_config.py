
import os
import numpy as np

def write_diffusion_jobs(
               sku='G1', 
               dataset='imagenette',
               caption_data='imagenet',
               caption_model='blip2',
               target_service='sing', 
               target_name='msroctows', 
               sla_tier='premium'):
    description = f'stable diffusion caption synthesize tasks vit gpt2'

    # write environment
    image = 'amlt-sing/pytorch-1.11.0-cuda11.6' #docker image

    setup = '  - python -m pip install -U pip\n' +\
            '  - pip install numpy\n' +\
            '  - pip install --upgrade torchvision \n' +\
            '  - pip install --upgrade torch \n' +\
            '  - pip install transformers\n' +\
            '  - pip install diffusers\n' +\
            '  - pip install tqdm\n' +\
            "  - pwd \n"
    #code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir = './'

    #storage:
    yaml_name = f'synthesize_{dataset}_{caption_model}'
    filepath = 'diffusion_caption_haoc'
    storage_account_name = 'ussclowpriv100data'
    data_container_name = 'jindwang'
    model_container_name = 'v-haochen2'
    os.makedirs(f'./amlt_config', exist_ok=True)
    with open(f'./amlt_config/' + yaml_name + '.yaml', 'w', encoding='utf-8') as w:
        w.write('description:'+' '+ description+'\n')
        w.write('target:'+'\n')
        w.write(' '+'service:'+' '+target_service+'\n')
        w.write(' '+'name:'+' '+target_name+'\n')
        if target_service == 'amlk8s':
            w.write(f' vc: resrchvc\n')
        elif target_service == 'sing':
            if target_name == 'msrresrchvc':
                workspace = 'msrresrchws'
            else:
                workspace = 'msroctows'
            w.write(f' workspace_name: {workspace}\n')
            

        # w.write(' '+'vc:'+' '+vc+'\n')
        w.write('environment:'+'\n')
        w.write(' '+'image:'+' '+image+'\n')
        w.write(' '+'setup:'+'\n'+setup+'\n')
        w.write('code:'+'\n')
        w.write(' '+'local_dir:'+' '+local_dir+'\n')
        w.write('storage:'+'\n')
        w.write(' '+filepath+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+model_container_name+'\n')
        w.write(' '+' '+'is_output:'+' '+'True'+'\n')
        w.write(' '+'data'+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+data_container_name+'\n')
        w.write(' '+' '+'mount_dir:'+' '+'/mnt/data'+'\n')
        w.write('jobs:'+'\n')


        guidance_list = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5]
        # caption_list = [True, False]
        if caption_model is not None:
            caption_list = [True]
        else:
            caption_list = [False]
        caption_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/caption_data/{caption_data}_caption_{caption_model}'
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230622/{dataset}'
        
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for use_caption in caption_list:
                
                data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                job_name = f'{dataset}_scale{guidance_scale}_caption{caption_model}'
                
                if use_caption:
                    command = f'python image_dataset_generate.py --dataset {dataset} --guidance_scale {guidance_scale} --use_caption --data_dir {data_dir} --caption_dir {caption_dir}'
                else:
                    command = f'python image_dataset_generate.py --dataset {dataset} --guidance_scale {guidance_scale} --data_dir {data_dir} --caption_dir {caption_dir}'

                jobs_list.append((job_name, command))
                job_cnt += 1
        
        jobs_list.sort()
        for job in jobs_list:
            job_name = job[0]
            command = job[1]
            w.write('- name:'+' '+job_name+'\n')
            w.write(' '+' sku:'+' '+f'{sku}'+'\n')
            if target_service == 'sing':
                w.write(' '+' sla_tier:'+' '+f'{sla_tier}'+'\n')
            w.write(' '+' command:'+'\n')
            w.write(' '+' -'+' '+command+'\n')
            w.write('\n')
        print(f"Generate {len(jobs_list)} tasks for {yaml_name}")


def write_diffusion_jobs_imagenet100(
               sku='G1', 
               dataset='imagenet100',
               caption_data='imagenet',
               caption_model='blip2',
               target_service='sing', 
               target_name='msroctows', 
               sla_tier='premium'):
    description = f'stable diffusion caption synthesize tasks'

    # write environment
    image = 'amlt-sing/pytorch-1.11.0-cuda11.6' #docker image

    setup = '  - python -m pip install -U pip\n' +\
            '  - pip install numpy\n' +\
            '  - pip install --upgrade torchvision \n' +\
            '  - pip install --upgrade torch \n' +\
            '  - pip install transformers\n' +\
            '  - pip install diffusers\n' +\
            '  - pip install tqdm\n' +\
            "  - pwd \n"
    #code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir = './'

    #storage:
    yaml_name = f'synthesize_{dataset}_{caption_model}'
    filepath = 'diffusion_caption_haoc'
    storage_account_name = 'ussclowpriv100data'
    data_container_name = 'jindwang'
    model_container_name = 'v-haochen2'
    os.makedirs(f'./amlt_config', exist_ok=True)
    with open(f'./amlt_config/' + yaml_name + '.yaml', 'w', encoding='utf-8') as w:
        w.write('description:'+' '+ description+'\n')
        w.write('target:'+'\n')
        w.write(' '+'service:'+' '+target_service+'\n')
        w.write(' '+'name:'+' '+target_name+'\n')
        if target_service == 'amlk8s':
            w.write(f' vc: resrchvc\n')
        elif target_service == 'sing':
            if target_name == 'msrresrchvc':
                workspace = 'msrresrchws'
            else:
                workspace = 'msroctows'
            w.write(f' workspace_name: {workspace}\n')
            

        # w.write(' '+'vc:'+' '+vc+'\n')
        w.write('environment:'+'\n')
        w.write(' '+'image:'+' '+image+'\n')
        w.write(' '+'setup:'+'\n'+setup+'\n')
        w.write('code:'+'\n')
        w.write(' '+'local_dir:'+' '+local_dir+'\n')
        w.write('storage:'+'\n')
        w.write(' '+filepath+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+model_container_name+'\n')
        w.write(' '+' '+'is_output:'+' '+'True'+'\n')
        w.write(' '+'data'+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+data_container_name+'\n')
        w.write(' '+' '+'mount_dir:'+' '+'/mnt/data'+'\n')
        w.write('jobs:'+'\n')


        guidance_list = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5]
        if caption_model is not None:
            caption_list = [True]
        else:
            caption_list = [False]
        caption_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/caption_data/{caption_data}_caption_{caption_model}'
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230622/{dataset}'

        data_piece_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for use_caption in caption_list:
                for data_piece in data_piece_list:
                    data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                    job_name = f'{dataset}_scale{guidance_scale}_caption{caption_model}_piece{data_piece}'
                    
                    if use_caption:
                        command = f'python image_dataset_generate_imagenet100.py --dataset {dataset} --guidance_scale {guidance_scale} --use_caption --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece}'
                    else:
                        command = f'python image_dataset_generate_imagenet100.py --dataset {dataset} --guidance_scale {guidance_scale} --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece}'

                    jobs_list.append((job_name, command))
                    job_cnt += 1
        
        jobs_list.sort()
        for job in jobs_list:
            job_name = job[0]
            command = job[1]
            w.write('- name:'+' '+job_name+'\n')
            w.write(' '+' sku:'+' '+f'{sku}'+'\n')
            if target_service == 'sing':
                w.write(' '+' sla_tier:'+' '+f'{sla_tier}'+'\n')
            w.write(' '+' command:'+'\n')
            w.write(' '+' -'+' '+command+'\n')
            w.write('\n')
        print(f"Generate {len(jobs_list)} tasks for {yaml_name}")


def write_diffusion_jobs_imagenet1k(
               sku='G1', 
               dataset='imagenet1k',
               caption_data='imagenet',
               caption_model='blip2',
               target_service='sing', 
               target_name='msroctows', 
               sla_tier='premium'):
    description = f'stable diffusion caption synthesize tasks'

    # write environment
    image = 'amlt-sing/pytorch-1.11.0-cuda11.6' #docker image

    setup = '  - python -m pip install -U pip\n' +\
            '  - pip install numpy\n' +\
            '  - pip install --upgrade torchvision \n' +\
            '  - pip install --upgrade torch \n' +\
            '  - pip install transformers\n' +\
            '  - pip install diffusers\n' +\
            '  - pip install tqdm\n' +\
            "  - pwd \n"
    #code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir = './'

    #storage:
    yaml_name = f'synthesize_{dataset}_blip2'
    filepath = 'diffusion_caption_haoc'
    storage_account_name = 'ussclowpriv100data'
    data_container_name = 'jindwang'
    model_container_name = 'v-haochen2'
    os.makedirs(f'./amlt_config', exist_ok=True)
    with open(f'./amlt_config/' + yaml_name + '.yaml', 'w', encoding='utf-8') as w:
        w.write('description:'+' '+ description+'\n')
        w.write('target:'+'\n')
        w.write(' '+'service:'+' '+target_service+'\n')
        w.write(' '+'name:'+' '+target_name+'\n')
        if target_service == 'amlk8s':
            w.write(f' vc: resrchvc\n')
        elif target_service == 'sing':
            if target_name == 'msrresrchvc':
                workspace = 'msrresrchws'
            else:
                workspace = 'msroctows'
            w.write(f' workspace_name: {workspace}\n')
            

        # w.write(' '+'vc:'+' '+vc+'\n')
        w.write('environment:'+'\n')
        w.write(' '+'image:'+' '+image+'\n')
        w.write(' '+'setup:'+'\n'+setup+'\n')
        w.write('code:'+'\n')
        w.write(' '+'local_dir:'+' '+local_dir+'\n')
        w.write('storage:'+'\n')
        w.write(' '+filepath+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+model_container_name+'\n')
        w.write(' '+' '+'is_output:'+' '+'True'+'\n')
        w.write(' '+'data'+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+data_container_name+'\n')
        w.write(' '+' '+'mount_dir:'+' '+'/mnt/data'+'\n')
        w.write('jobs:'+'\n')


        # guidance_list = [1.5, 2]
        guidance_list = [1.5]
        # caption_list = [True, False]
        caption_list = [True, False]
        caption_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/caption_data/{caption_data}_caption_{caption_model}'
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230622/{dataset}'

        # data_piece_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        data_piece_list = np.arange(100)
        
        
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for use_caption in caption_list:
                for data_piece in data_piece_list:
                    data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                    job_name = f'{dataset}_scale{guidance_scale}_caption{caption_model}_piece{data_piece}'
                    
                    if use_caption:
                        command = f'python image_dataset_generate_imagenet1k.py --dataset {dataset} --guidance_scale {guidance_scale} --use_caption --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece}'
                    else:
                        command = f'python image_dataset_generate_imagenet1k.py --dataset {dataset} --guidance_scale {guidance_scale} --data_dir {data_dir} --caption_dir {caption_dir} --data_piece {data_piece}'

                    jobs_list.append((job_name, command))
                    job_cnt += 1
        
        jobs_list.sort()
        for job in jobs_list:
            job_name = job[0]
            command = job[1]
            w.write('- name:'+' '+job_name+'\n')
            w.write(' '+' sku:'+' '+f'{sku}'+'\n')
            if target_service == 'sing':
                w.write(' '+' sla_tier:'+' '+f'{sla_tier}'+'\n')
            w.write(' '+' command:'+'\n')
            w.write(' '+' -'+' '+command+'\n')
            w.write('\n')
        print(f"Generate {len(jobs_list)} tasks for {yaml_name}")


def write_llm_rewrite_jobs(
               sku='G1', 
               caption_data='imagenet',
               caption_model='blip2',
               llm_model='vicuna-7b-v1.3',
               port=9008,
               target_service='sing', 
               target_name='msroctows', 
               sla_tier='premium'):
    description = f'vicuna caption rewrite tasks'

    # write environment
    image = 'amlt-sing/pytorch-1.11.0-cuda11.6' #docker image

    setup = '  - python -m pip install -U pip\n' +\
            '  - pip install numpy\n' +\
            '  - pip install transformers\n' +\
            '  - pip install diffusers\n' +\
            '  - pip install tqdm\n' +\
            '  - pip install fschat\n' +\
            '  - pip install deepspeed\n' +\
            '  - pip install openai\n' +\
            '  - pip install typing-extensions==4.5.0\n' +\
            "  - pwd \n"
    #code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir = './'    
    n_gpus_for_one_dataset = 200
    num_gpus = int(sku[-1])
    data_piece_list = np.arange(n_gpus_for_one_dataset)

    #storage:
    yaml_name = f'llm_rewrite_{caption_data}_{caption_model}'
    filepath = 'diffusion_caption_haoc'
    storage_account_name = 'ussclowpriv100data'
    data_container_name = 'jindwang'
    model_container_name = 'v-haochen2'
    os.makedirs(f'./amlt_config', exist_ok=True)
    with open(f'./amlt_config/' + yaml_name + '.yaml', 'w', encoding='utf-8') as w:
        w.write('description:'+' '+ description+'\n')
        w.write('target:'+'\n')
        w.write(' '+'service:'+' '+target_service+'\n')
        w.write(' '+'name:'+' '+target_name+'\n')
        if target_service == 'amlk8s':
            w.write(f' vc: resrchvc\n')
        elif target_service == 'sing':
            if target_name == 'msrresrchvc':
                workspace = 'msrresrchws'
            else:
                workspace = 'msroctows'
            w.write(f' workspace_name: {workspace}\n')
            

        # w.write(' '+'vc:'+' '+vc+'\n')
        w.write('environment:'+'\n')
        w.write(' '+'image:'+' '+image+'\n')
        w.write(' '+'setup:'+'\n'+setup+'\n')
        w.write('code:'+'\n')
        w.write(' '+'local_dir:'+' '+local_dir+'\n')
        w.write('storage:'+'\n')
        w.write(' '+filepath+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+model_container_name+'\n')
        w.write(' '+' '+'is_output:'+' '+'True'+'\n')
        w.write(' '+'data'+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+data_container_name+'\n')
        w.write(' '+' '+'mount_dir:'+' '+'/mnt/data'+'\n')
        w.write('jobs:'+'\n')


        n_gpus_for_one_dataset = 200
        num_gpus = int(sku[-1])
        data_piece_list = np.arange(n_gpus_for_one_dataset)
        caption_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/caption_data/{caption_data}_caption_{caption_model}'
        base_save_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/caption_data/llm_rewrite/20230624/{caption_data}_caption_{caption_model}_llm_rewrite_raw'

        jobs_list = []
        for data_piece in data_piece_list:
            job_name = f'llm_rewrite_{caption_data}_{caption_model}_{data_piece}'
            command_0 = 'nohup python3 -m fastchat.serve.controller > controller.log &' 
            command_1 = f'nohup python3 -m fastchat.serve.model_worker --model-path lmsys/{llm_model} --num-gpus {num_gpus} > model_worker.log &'
            command_2 = f'sleep 500'
            command_3 = f'nohup python3 -m fastchat.serve.openai_api_server --host localhost --port {port} > api_server.log &'
            command_4 = f'sleep 10'
            command_5 = f'python3 llm_rewrite.py --caption-dir {caption_dir} --save-dir {base_save_dir} --llm-per-caption 5 --llm-tempreture 0.5 --port {port} --model {llm_model} --n_gpus_for_one_dataset {n_gpus_for_one_dataset} --data_piece {data_piece}'
            jobs_list.append((job_name, [command_0, command_1, command_2, command_3, command_4, command_5]))
        
        for job in jobs_list:
            job_name = job[0]
            command_list = job[1]
            w.write('- name:'+' '+job_name+'\n')
            w.write(' '+' sku:'+' '+f'{sku}'+'\n')
            if target_service == 'sing':
                w.write(' '+' sla_tier:'+' '+f'{sla_tier}'+'\n')
            w.write(' '+' command:'+'\n')
            for command in command_list:
                w.write(' '+' -'+' '+command+'\n')
            w.write('\n')
        print(f"Generate {len(jobs_list)} tasks for {yaml_name}")


def write_train_jobs(
               sku='G1', 
               dataset='imagenette',
               target_service='sing', 
               target_name='msroctows', 
               sla_tier='premium'):
    description = f'stable diffusion caption train tasks'

    # write environment
    image = 'amlt-sing/pytorch-1.11.0-cuda11.6' #docker image

    setup = '  - python -m pip install -U pip\n' +\
            '  - pip install numpy\n' +\
            '  - pip install --upgrade torchvision \n' +\
            '  - pip install --upgrade torch \n' +\
            '  - pip install transformers\n' +\
            '  - pip install diffusers\n' +\
            "  - pwd \n"
    #code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir = './'

    #storage:
    yaml_name = f'train_{dataset}'
    filepath = 'diffusion_caption_haoc'
    storage_account_name = 'ussclowpriv100data'
    data_container_name = 'jindwang'
    model_container_name = 'v-haochen2'
    os.makedirs(f'./amlt_config', exist_ok=True)

    with open(f'./amlt_config/' + yaml_name + '.yaml', 'w', encoding='utf-8') as w:
        w.write('description:'+' '+ description+'\n')
        w.write('target:'+'\n')
        w.write(' '+'service:'+' '+target_service+'\n')
        w.write(' '+'name:'+' '+target_name+'\n')
        if target_service == 'amlk8s':
            w.write(f' vc: resrchvc\n')
        elif target_service == 'sing':
            if target_name == 'msrresrchvc':
                workspace = 'msrresrchws'
            else:
                workspace = 'msroctows'
            w.write(f' workspace_name: {workspace}\n')
            

        # w.write(' '+'vc:'+' '+vc+'\n')
        w.write('environment:'+'\n')
        w.write(' '+'image:'+' '+image+'\n')
        w.write(' '+'setup:'+'\n'+setup+'\n')
        w.write('code:'+'\n')
        w.write(' '+'local_dir:'+' '+local_dir+'\n')
        w.write('storage:'+'\n')
        w.write(' '+filepath+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+model_container_name+'\n')
        w.write(' '+' '+'is_output:'+' '+'True'+'\n')
        w.write(' '+'data'+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+data_container_name+'\n')
        w.write(' '+' '+'mount_dir:'+' '+'/mnt/data'+'\n')
        w.write('jobs:'+'\n')


        guidance_list = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5]
        # caption_model_list = ['blip2', 'vit-gpt2', 'syn_blip2', None]
        # base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230622/{dataset}'
        caption_model_list = ['syn_gs_1_blip2', 'syn_gs_1.5_blip2', 'syn_gs_2_blip2', 'syn_gs_2.5_blip2', 'syn_gs_3_blip2', 'syn_gs_4_blip2', 'syn_gs_5_blip2', 'syn_gs_6_blip2', 'syn_gs_7.5_blip2']
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230622/imagenette'
        base_save_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train_models/20230622/imagenette'
        batch_size = 128 
        
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for caption_model in caption_model_list:
                

                data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                save_dir = base_save_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                job_name = f'{dataset}_scale{guidance_scale}_caption{caption_model}'
                
                if caption_model is not None:
                    use_caption = True
                else:
                    use_caption = False
                
                if use_caption:
                    # data_dir = data_dir + f'/' + dataset + "_use_caption_gs_" + str(guidance_scale)
                    command = f'python eval_dataset.py --dataset {dataset} --guidance_scale {guidance_scale} --use_caption --data_path {data_dir} --batch_size {batch_size} --save_dir {save_dir}'
                else:
                    # data_dir = data_dir + f'/' + dataset + "_gs_" + str(guidance_scale)
                    command = f'python eval_dataset.py --dataset {dataset} --guidance_scale {guidance_scale} --data_path {data_dir} --batch_size {batch_size} --save_dir {save_dir}'

                jobs_list.append((job_name, command))
                job_cnt += 1
        
        jobs_list.sort()
        for job in jobs_list:
            job_name = job[0]
            command = job[1]
            w.write('- name:'+' '+job_name+'\n')
            w.write(' '+' sku:'+' '+f'{sku}'+'\n')
            if target_service == 'sing':
                w.write(' '+' sla_tier:'+' '+f'{sla_tier}'+'\n')
            w.write(' '+' command:'+'\n')
            w.write(' '+' -'+' '+command+'\n')
            w.write('\n')
        print(f"Generate {len(jobs_list)} tasks for {yaml_name}")



def write_train_jobs_imagenet100(
               sku='G1', 
               dataset='imagenet100',
               target_service='sing', 
               target_name='msroctows', 
               sla_tier='premium'):
    description = f'stable diffusion caption train tasks'

    # write environment
    image = 'amlt-sing/pytorch-1.11.0-cuda11.6' #docker image

    setup = '  - python -m pip install -U pip\n' +\
            '  - pip install numpy\n' +\
            '  - pip install --upgrade torchvision \n' +\
            '  - pip install --upgrade torch \n' +\
            '  - pip install transformers\n' +\
            '  - pip install diffusers\n' +\
            "  - pwd \n"
    #code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir = './'

    #storage:
    yaml_name = f'train_{dataset}'
    filepath = 'diffusion_caption_haoc'
    storage_account_name = 'ussclowpriv100data'
    data_container_name = 'jindwang'
    model_container_name = 'v-haochen2'
    os.makedirs(f'./amlt_config', exist_ok=True)

    with open(f'./amlt_config/' + yaml_name + '.yaml', 'w', encoding='utf-8') as w:
        w.write('description:'+' '+ description+'\n')
        w.write('target:'+'\n')
        w.write(' '+'service:'+' '+target_service+'\n')
        w.write(' '+'name:'+' '+target_name+'\n')
        if target_service == 'amlk8s':
            w.write(f' vc: resrchvc\n')
        elif target_service == 'sing':
            if target_name == 'msrresrchvc':
                workspace = 'msrresrchws'
            else:
                workspace = 'msroctows'
            w.write(f' workspace_name: {workspace}\n')
            

        # w.write(' '+'vc:'+' '+vc+'\n')
        w.write('environment:'+'\n')
        w.write(' '+'image:'+' '+image+'\n')
        w.write(' '+'setup:'+'\n'+setup+'\n')
        w.write('code:'+'\n')
        w.write(' '+'local_dir:'+' '+local_dir+'\n')
        w.write('storage:'+'\n')
        w.write(' '+filepath+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+model_container_name+'\n')
        w.write(' '+' '+'is_output:'+' '+'True'+'\n')
        w.write(' '+'data'+':'+'\n')
        w.write(' '+' '+'storage_account_name:'+' '+storage_account_name+'\n')
        w.write(' '+' '+'container_name:'+' '+data_container_name+'\n')
        w.write(' '+' '+'mount_dir:'+' '+'/mnt/data'+'\n')
        w.write('jobs:'+'\n')


        guidance_list = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5]
        caption_model_list = ['blip2', 'vit-gpt2', 'syn_blip2', None]
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230622/{dataset}'
        base_save_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train_models/20230622/{dataset}'
        batch_size = 512 
        
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for caption_model in caption_model_list:
                
                
                data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                save_dir = base_save_dir + f'/{dataset}_scale{guidance_scale}_caption{caption_model}'
                job_name = f'{dataset}_scale{guidance_scale}_caption{caption_model}'
                
                if caption_model is not None:
                    use_caption = True
                else:
                    use_caption = False
                
                
                if use_caption:
                    # data_dir = data_dir + f'/' + dataset + "_use_caption_gs_" + str(guidance_scale)
                    command = f'CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 eval_dataset_distributed.py --dataset {dataset} --guidance_scale {guidance_scale} --use_caption --data_path {data_dir} --batch_size {batch_size} --save_dir {save_dir}'
                else:
                    # data_dir = data_dir + f'/' + dataset + "_gs_" + str(guidance_scale)
                    command = f'CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 eval_dataset_distributed.py --dataset {dataset} --guidance_scale {guidance_scale} --data_path {data_dir} --batch_size {batch_size} --save_dir {save_dir}'

                jobs_list.append((job_name, command))
                job_cnt += 1
        
        jobs_list.sort()
        for job in jobs_list:
            job_name = job[0]
            command = job[1]
            w.write('- name:'+' '+job_name+'\n')
            w.write(' '+' sku:'+' '+f'{sku}'+'\n')
            if target_service == 'sing':
                w.write(' '+' sla_tier:'+' '+f'{sla_tier}'+'\n')
            w.write(' '+' command:'+'\n')
            w.write(' '+' -'+' '+command+'\n')
            w.write('\n')
        print(f"Generate {len(jobs_list)} tasks for {yaml_name}")


if __name__ == '__main__':
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='blip2')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='australia1GPUcl', dataset='imagenette', caption_model='syn_blip2', caption_data='imagenet_gs_1.5')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='vit-gpt2')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model=None)
    
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_1_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_1.5_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_2_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_2.5_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_3_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_4_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_5_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_6_blip2', caption_data='imagenette')
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette', caption_model='syn_gs_7.5_blip2', caption_data='imagenette')
    
    # TODO: llm rewrite
    
    
    write_diffusion_jobs_imagenet100(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenet100', caption_model='blip2')
    write_diffusion_jobs_imagenet100(sku='G1', target_service='aml', target_name='australia1GPUcl', dataset='imagenet100', caption_model='syn_blip2', caption_data='imagenet_gs_1.5')
    write_diffusion_jobs_imagenet100(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenet100', caption_model='vit-gpt2')
    write_diffusion_jobs_imagenet100(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenet100', caption_model=None)
    
    # write_diffusion_jobs_imagenet1k(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenet1k')
    
    # write_train_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette')
    # write_train_jobs_imagenet100(sku='G4', target_service='aml', target_name='canadav100cl', dataset='imagenet100', caption='blip')
    
    write_train_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette')
    write_train_jobs_imagenet100(sku='G4', target_service='aml', target_name='canadav100cl', dataset='imagenet100')
    
    
    write_llm_rewrite_jobs(sku='G4', target_service='sing', target_name='canadav100cl', caption_data='imagenet', caption_model='blip2', llm_model='vicuna-13b-v1.3', port=9008)