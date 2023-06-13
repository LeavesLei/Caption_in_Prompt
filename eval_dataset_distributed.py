import os
import torch
import torch.distributed as dist
import time
from network import ResNet50
from utils import train_distributed, test_distributed, get_hms
from load_data import get_dataset, get_class_name, get_curated_dataset
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imagenet1k", help="dataset")
parser.add_argument("--guidance_scale", type=float, default=1.5, help="guidance scale")
parser.add_argument("--data_path", type=str, default="/media/slei/slei_disk/data/curated_imagenet", help="data path")
parser.add_argument("--caption", type=str, default="none", help="type of caption")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
# parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

# Print Args
print("--------args----------")
for k in list(vars(args).keys()):
    print("%s: %s" % (k, vars(args)[k]))
print("--------args----------\n")

# ImageNetPath = '/media/slei/slei_disk/data/ImageNet'I
ImageNetPath = '/media/Bootes/datasets/imagenet'

dataset = args.dataset
guidance_scale = args.guidance_scale
data_path = args.data_path
caption = args.caption
batch_size = args.batch_size
local_rank = int(os.environ["LOCAL_RANK"])

nprocs = torch.cuda.device_count()
batch_size = int(batch_size / nprocs)
dist.init_process_group(backend="nccl")

print("nprocs: %s"%{nprocs})
print("normalized batch size: %s"%{batch_size})

if caption=='none':
    dataset_name = dataset + "_no-caption_gs_" + str(guidance_scale)
elif 'blip' in caption:
    dataset_name = dataset + "_blip_gs_" + str(guidance_scale)
elif 'vit' in caption:
    dataset_name = dataset + "_vit-gpt_gs_" + str(guidance_scale)
else:
    raise ValueError("Wrong caption value")

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

# Network init
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channel = 3
num_classes = 1000
net = ResNet50(channel, num_classes)
torch.cuda.set_device(local_rank)
net.cuda(local_rank)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])


# load real data
channel, im_size, num_classes, class_names, real_dst_train, dst_test, testloader = get_dataset(data_path=ImageNetPath, batch_size=1, subset=dataset)

# load curated datasets
class_names = get_class_name(dataset)
dst_train = get_curated_dataset(dataset_name, data_path, class_names, 1.)

train_sampler = torch.utils.data.distributed.DistributedSampler(dst_train)
trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, sample=train_sampler)

# Train
lr =0.1
start_epoch = 1
num_epochs = 200


elapsed_time = 0
for epoch in range(start_epoch, start_epoch + num_epochs):

    start_time = time.time()

    train_sampler.set_epoch(epoch)

    train_distributed(net=net, trainloader=trainloader, local_rank=local_rank, epoch=epoch, lr=lr)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print("| Elapsed time : %d:%02d:%02d" % (get_hms(elapsed_time)))

# Test
print("Test accuracy: ")
print(test_distributed(net, testloader, local_rank, 1))

if local_rank == 0:
    torch.save(net, 'models/resnet50_' + dataset_name)