guidance_scale=1 # [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5]

use_caption=False

dataset='imagenette'

data_dir=PATH_TO_SAVE_GENERATED_DATA

caption_dir=PATH_OF_IMAGE_CAPTION

batch_size=64


if [ $use_caption ]
then
    python image_dataset_generate.py --dataset $dataset --guidance_scale $guidance_scale --use_caption --data_dir $data_path --caption_dir $caption_dir
    python eval_dataset --dataset $dataset --guidance_scale $guidance_scale --use_caption --data_path $data_dir --batch_size $batch_size
else
    python image_dataset_generate.py --dataset $dataset --guidance_scale $guidance_scale --data_dir $data_path --caption_dir $caption_dir
    python eval_dataset --dataset $dataset --guidance_scale $guidance_scale --data_path $data_dir --batch_size $batch_size
fi