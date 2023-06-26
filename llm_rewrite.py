import os
import re
import argparse
import openai
from load_data import get_class_name, get_class_index, get_id_class_name_map_dict
from utils import load_list, save_list
from tqdm import tqdm
openai.api_key = "EMPTY" # Not support yet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--caption-dir', type=str, default='caption_data/imagenet_caption_blip2')
    parser.add_argument('--save-dir', type=str, default='caption_data/imagenet_caption_blip2_llm_rewrite')
    parser.add_argument('--llm-per-caption', type=int, default=5, help='number of caption candidates refined from LLM')
    parser.add_argument('--llm-tempreture', type=float, default=1.0, help='LLM tempreture')
    parser.add_argument('--port', type=int, default=9001, help='LLM tempreture')
    parser.add_argument("--n_gpus_for_one_dataset", type=int, default=100, help="xxx")
    parser.add_argument("--data_piece", type=int, default=0, help="xxx")
    parser.add_argument('--model', type=str, default='vicuna-13b-v1.3')
    args = parser.parse_args()
    return args


def process_raw_text(raw_text):
    split_text = raw_text.split('#Answer')

    # Make sure the #Answer section exists
    if len(split_text) > 1:
        # Use a regular expression to find all items under #Answer
        answers = re.findall('\d+\.\s(.*?)\n|\s(.*?)\n', split_text[1])
        refine_answers = []
        for ans in answers:
            refine_answers.append(max(ans, key = len))
        return refine_answers
    else:
        return []


def main(args): 
    openai.api_base = f"http://localhost:{args.port}/v1"
    
    class_names = get_class_name('imagenet1k')
    data_piece = args.data_piece
    n_gpus_for_one_dataset = args.n_gpus_for_one_dataset
    n_divide = len(class_names) // n_gpus_for_one_dataset
    class_names = class_names[data_piece * n_divide: (data_piece + 1) * n_divide]
    dict_id_to_class_name, dict_class_name_to_id = get_id_class_name_map_dict()
    
    caption_name_list = [dict_class_name_to_id[i] for i in class_names]
    # caption_name_list = os.listdir(args.caption_dir)
    caption_path_list = [os.path.join(args.caption_dir, f) for f in caption_name_list]
    
    model = args.model
    num_rewrite_per_caption = args.llm_per_caption
    tempreture = args.llm_tempreture
    save_dir = args.save_dir 
    save_dir = save_dir + f'{model}_temp{tempreture}_num{num_rewrite_per_caption}'
    os.makedirs(save_dir, exist_ok=True)
    raw_output_path = os.path.join(save_dir, f'raw_output_{data_piece}.txt')
    with open(raw_output_path, 'a') as f:
        f.write('start rewrite\n')
    
    start_idx, start_j = 0, 0
    re_startidx = r"Generate (([0-9]|\.)*) class"
    re_startj =  r"class (([0-9]|\.)*) caption"
    try:
        with open(raw_output_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('Generate'):
                    start_idx = int(re.search(re_startidx, line).group(1))
                    start_j = int(re.search(re_startj, line).group(1))
    except:
        pass
    print(f"start idx {start_idx}, start j {start_j}")
    
    # category = 'tuna'
    # caption = 'A photo of tuna. A tuna on the beach.'
    for idx, class_name in tqdm(enumerate(class_names), total=len(class_names)):
        
        if idx < start_idx:
            print(f"skip {idx} class")
            continue
        
        caption_path = caption_path_list[idx]
        caption_list = load_list(caption_path)
        caption_list = list(set(caption_list))

        rewrite_caption_list = []
        for j, caption in tqdm(enumerate(caption_list), total=len(caption_list)):
        
            if j < start_j:
                print(f"skip {j} caption")
                continue
        
            print(f"Generate {class_name}: {caption_list[0]}\n")
            start_text = f"Generate {idx} class {j} caption"
            with open(raw_output_path, 'a') as f:
                    f.write(start_text + '\n')
        
            prompt = f"This is the image caption about {class_name} category, please refine and rewrite it to {num_rewrite_per_caption} more diverse and informative caption candidates.\n" +\
                     f"#Caption\n{caption}\n" +\
                      "#Answer\n"
            
            # create a completion
            try:
                completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=512, tempreture=tempreture)
                # print the completion
                raw_output = prompt + completion.choices[0].text
                print(raw_output)
                print('\n')

                with open(raw_output_path, 'a') as f:
                    f.write(raw_output + '\n')
            except:
                continue
            

if __name__ == '__main__':
    args = parse_args()
    main(args)
    