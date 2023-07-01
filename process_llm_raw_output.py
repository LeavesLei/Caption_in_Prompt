import re
import os
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from utils import save_list
from load_data import get_class_name, get_class_index, get_id_class_name_map_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-dir', default='./caption_data/llm_rewrite/20230624/imagenet_caption_blip2_llm_rewrite_rawvicuna-13b-v1.3_temp0.5_num5', type=str)
    parser.add_argument('--save-dir', default='./caption_data', type=str)
    parser.add_argument('--n-caps-per-cls', type=int, default=1300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def load_raw_data(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            data.append(line)
    return data
    

# func for removing emoji, backslash, and specifc characters (e.g. 「」“”【】♂️♀️)
def filter_emoji(desstr, restr=''):
    res = re.compile(u'\s*[\U00010000-\U0010ffff\\uD800-\\uDBFF\\uDC00-\\uDFFF]\s*')
    return re.sub(r'[\\「」“”【】♂️♀️]', "", res.sub(restr, desstr))


def check_match_item(line_pt, data):
    if re.match(r"Generate [0-9]+ class [0-9]+ caption", data[line_pt]) and re.match(r"This is the image caption about [\w\s(+*)'-\.]+ category, please refine and rewrite it to [0-9] more diverse and informative caption candidates.", data[line_pt+1]):
        return True
    else:
        return False

def find_next_match(line_pt, data):
    while line_pt < len(data) - 1:
        if check_match_item(line_pt, data):
            break
        else:
            line_pt += 1
    return line_pt


def remove_prefixes(text):
    # match prefixes like '1.', '(1)', 'A', 'a)', '*', '**', '* ', '-'
    pattern = r"^\s*(\d+\.\s*|\(\d+\)\s*|[A-Za-z]\)\s*|\*\*?\s*|-)\s*"
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def remove_edge_quotes(text):
    # Remove straight double quotes from the beginning and the end
    text = text.lstrip('"').rstrip('"')
    # Remove straight single quotes from the beginning and the end
    text = text.lstrip('\'').rstrip('\'')
    # Remove opening and closing curly quotes from the beginning and the end
    text = text.lstrip('“‘').rstrip('”’')
    return text


def extract_caption(line):
    # 正则表达式: ^([0-9a-zA-Z-"].|\(?[0-9a-zA-Z]\)|\*).*["0-9a-zA-Z].*
    # ^只匹配开头
    # 开头为序号（e.g. 1. 2. 3. 4. 5.）
    # 开头为字母 (e.g. A. B. C. D. E.)
    # 开头为括号加数字 (e.g. (1) (2) (3) (4) (5))
    # 开头为半括号加字母 (e.g. a) b) c) d) e))
    # 开头为 * or *[空格]
    # 开头为 -
    # 开头为 "
    # 开头没有上述标识符，直接以单词开头
    # 只rewrite了一条的caption drop掉
    # 非英文caption全部drop
    
    # some edge cases
    if line.startswith('#Answer') or line.startswith('#Caption') or line.startswith('Please'):
        return None
    
    # filter emoji 
    line = filter_emoji(line)
    
    # remove item starting items
    line = remove_prefixes(line)
    
    # remove edge quotation
    line = remove_edge_quotes(line)
    
    # detect is english
    if not line.isascii():
        return None
    
    # drop 过短以及过长的 caption (5<length<30)，另外LLM输出有可能单词全连在一起（e.g. Thehammerheadsharkisoneofthemostelegantswimmersofthesea,glidingthroughth waterwitheaseandelegance.）
    if len(line.strip().split(' ')) > 5 and len(line.strip().split(' ')) < 65: 
        return line
    else:
        return None


def process_raw_data(data):
    data_list = []
    curr_line_pt = find_next_match(0, data)
    while curr_line_pt < len(data) - 1:
        # find the next match
        next_line_pt = find_next_match(curr_line_pt + 1, data)
        
        # extract category name 
        category_name = re.findall(r"This is the image caption about ([\w\s(+*)'-\.]+) category", data[curr_line_pt+1])[0]
        item_dict = {'class': category_name}
        
        # extract caption
        if re.match(r'#Caption', data[curr_line_pt+2]):
            caption = data[curr_line_pt+3]
            item_dict['caption'] = caption
        
        # extract answer
        temp = [] 
        if re.match(r'#Answer', data[curr_line_pt+4]):
            answer_section = data[curr_line_pt+4:next_line_pt]
            temp_line_pt = curr_line_pt + 5 
            while temp_line_pt < next_line_pt:
                rewrite_caption = extract_caption(data[temp_line_pt])
                if rewrite_caption is not None:
                    temp.append(rewrite_caption)
                temp_line_pt += 1
        
        # go next 
        curr_line_pt = next_line_pt
        
        if len(temp) == 0:
            continue
        
        item_dict['llm_rewrite'] = temp[:5]
        data_list.append(item_dict)
    return data_list


def main(args):
    raw_data_dir = args.raw_data_dir
    save_dir = args.save_dir 
    n_caps_per_cls = args.n_caps_per_cls
    seed = args.seed 
    
    rewrite_caption_data = raw_data_dir.split('/')[-1].split('_')
    rewrite_caption_data = rewrite_caption_data[:rewrite_caption_data.index('llm')]
    save_dir = os.path.join(save_dir, '_'.join(rewrite_caption_data) + f'_n{n_caps_per_cls}_s{seed}')
    os.makedirs(save_dir, exist_ok=True)
    
    # load all text files in raw data folder
    raw_files = sorted(glob(os.path.join(raw_data_dir, '*.txt')))
    
    # process each file
    all_processed_data = []
    for idx, raw_file in tqdm(enumerate(raw_files), total=len(raw_files)):
        raw_data = load_raw_data(raw_file)
        processed_data = process_raw_data(raw_data)
        all_processed_data.extend(processed_data)
    
    # save rewrite captions according to classes
    class_names = get_class_name('imagenet1k')
    dict_class_id_to_name, dict_class_name_to_id = get_id_class_name_map_dict()
    class_ids = [dict_class_name_to_id[class_name] for class_name in class_names]
    
    # map processed data to class:caption_list dict
    class2rewrite = {class_id: [] for class_id in class_ids}
    class2orig =  {class_id: [] for class_id in class_ids}
    for item_dict in all_processed_data:
        class_id = dict_class_name_to_id[item_dict['class']]
        class2rewrite[class_id].extend(item_dict['llm_rewrite'])
        class2orig[class_id].append(item_dict['caption'])
    
    # debug check
    rewrite_cnt = [len(item) for class_id, item in class2rewrite.items()]
    orig_cnt = [len(item) for class_id, item in class2orig.items()]
    print(rewrite_cnt)
    print(orig_cnt)
    
    empty_names = []
    for class_id, item in class2rewrite.items():
        if len(item) <= 100:
            class_name = dict_class_id_to_name[class_id]
            empty_names.append(class_name)
    print(len(empty_names))
    print(empty_names)
    
    # save caption_data
    np.random.seed(args.seed)
    for class_name, rewrite_caption_list in class2rewrite.items():
        np.random.shuffle(rewrite_caption_list)
        rewrite_caption_list = rewrite_caption_list[:args.n_caps_per_cls]
        class_id = dict_class_id_to_name[class_name]
        save_list(rewrite_caption_list, os.path.join(save_dir, class_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)