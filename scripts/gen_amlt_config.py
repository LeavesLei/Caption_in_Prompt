
import os

def write_diffusion_jobs(
               sku='G1', 
               dataset='imagenette',
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
    yaml_name = f'synthesize_{dataset}'
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
        caption_list = [True, False]
        caption_dir = '/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/caption_data/imagenet_caption_blip2'
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230607'
        
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for use_caption in caption_list:
                
                data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{use_caption}'
                job_name = f'{dataset}_scale{guidance_scale}_caption{use_caption}'
                
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
    yaml_name = f'synthesize_{dataset}'
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


        guidance_list = [1.5, 2]
        caption_list = [True, False]
        caption_dir = '/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/caption_data/imagenet_caption_blip2'
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230608'

        data_piece_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for use_caption in caption_list:
                for data_piece in data_piece_list:
                    data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{use_caption}'
                    job_name = f'{dataset}_scale{guidance_scale}_caption{use_caption}_piece{data_piece}'
                    
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
        caption_list = [True, False]
        base_data_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train/20230607'
        base_save_dir = f'/mnt/diffusion_caption_haoc/projects/diffusion_caption_train_models/20230608'
        batch_size = 128 
        
        jobs_list = []
        job_cnt = 0
        for guidance_scale in guidance_list:
            for use_caption in caption_list:
                
                data_dir = base_data_dir + f'/{dataset}_scale{guidance_scale}_caption{use_caption}'
                save_dir = base_save_dir + f'/{dataset}_scale{guidance_scale}_caption{use_caption}'
                job_name = f'{dataset}_scale{guidance_scale}_caption{use_caption}'
                
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




if __name__ == '__main__':
    write_diffusion_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette')
    write_diffusion_jobs_imagenet100(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenet100')
    write_train_jobs(sku='G1', target_service='aml', target_name='canada1GPUcl', dataset='imagenette')