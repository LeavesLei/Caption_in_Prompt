import torch
import time
from network import ResNet50
from utils import train, test, get_hms
from load_data import get_dataset, get_class_name, get_curated_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imagenette", help="dataset")
parser.add_argument("--guidance_scale", type=float, default=3, help="guidance scale")
parser.add_argument("--data_path", type=str, default="/media/slei/slei_disk/data/curated_imagenet", help="data path")
parser.add_argument('--use_caption', action='store_true')
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
args = parser.parse_args()

# Print Args
print("--------args----------")
for k in list(vars(args).keys()):
    print("%s: %s" % (k, vars(args)[k]))
print("--------args----------\n")

ImageNetPath = '/media/slei/slei_disk/data/ImageNet'

dataset = args.dataset
guidance_scale = args.guidance_scale
data_path = args.data_path
use_caption = args.use_caption
batch_size = args.batch_size


if use_caption:
    dataset_name = dataset + "_use_caption_gs_" + str(guidance_scale)
else:
    dataset_name = dataset + "_gs_" + str(guidance_scale)

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

# load real data
channel, im_size, num_classes, class_names, real_dst_train, dst_test, testloader = get_dataset(data_path=ImageNetPath, batch_size=1, subset=dataset)

# load curated datasets
class_names = get_class_name(dataset)
data_path = "/media/slei/slei_disk/data/curated_imagenet"
dst_train = get_curated_dataset(dataset_name, data_path, class_names, 1.)
trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=2)

# Network init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet50(channel, num_classes).to(device)

# Train
lr =0.1
start_epoch = 1
num_epochs = 200


elapsed_time = 0
for epoch in range(start_epoch, start_epoch + num_epochs):

    start_time = time.time()
    train(net=net, trainloader=trainloader, epoch=epoch, lr=lr)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print("| Elapsed time : %d:%02d:%02d" % (get_hms(elapsed_time)))

# Test
print(test(net, testloader, 1))

torch.save(net, 'models/resnet50_' + dataset_name + '_caption' + str(use_caption))
