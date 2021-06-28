import argparse
import os
from PIL import Image
import torchvision.transforms as tf
from src.data.dataset import CustomDataset
from torch.utils.data import DataLoader
import torchvision.utils as utils

import torch
from src.model.generator import Generator
from src.model.discriminator import Discriminator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='CycleGAN_Train')
parser.add_argument('--imga_path', type=str, default=None, help='путь до изображения А')
parser.add_argument('--imgb_path', type=str, default=None, help='путь до изображения B')
parser.add_argument('--weight_path', type=str, default=None, help='путь до весов модели')
parser.add_argument('--task', type=str, default='horse2zebra', help='задача, для которой выбираются веса ['
                                                                    'horse2zebra, person2avatar]')
parser.add_argument('--images', type=str, default='AB', help='какие фейки будут сохранены (если загружается фото А, '
                                                             'то фейк B и наоборот) [AB, A, B]')

args = parser.parse_args()

# сеть
genA = Generator(3, 3).to(DEVICE)
genB = Generator(3, 3).to(DEVICE)

if args.weight_path is None:
    if args.task == 'horse2zebra':
        if os.path.exists('/h2zCycleGAN.pth'):
            checkpoint = torch.load('/h2zCycleGAN.pth')
            genA.load_state_dict(checkpoint['genA_state_dict'])
            genB.load_state_dict(checkpoint['genB_state_dict'])

        else:
            assert 'Файл весов отсутсвтует'
    elif args.task == 'person2avatar':
        if os.path.exists('/p2aCycleGAN.pth'):
            checkpoint = torch.load('/h2zCycleGAN.pth')
            genA.load_state_dict(checkpoint['genA_state_dict'])
            genB.load_state_dict(checkpoint['genB_state_dict'])

        else:
            assert 'Файл весов отсутсвтует'

else:
    checkpoint = torch.load(args.weight_path)
    genA.load_state_dict(checkpoint['genA_state_dict'])
    genB.load_state_dict(checkpoint['genB_state_dict'])

transform = tf.Compose(
    [tf.ToTensor(), tf.Resize((256, 256)), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

directory = '/horse2zebra' if args.task == 'horse2zebra' else '/person2avatar'
dataset = CustomDataset(directory, 'test')
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

imgA = None
imgB = None
fakeA = None
fakeB = None

folder_path = '/test_results'
for file_object in os.listdir(folder_path):
    file_object_path = os.path.join(folder_path, file_object)
    if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
        os.unlink(file_object_path)


if args.images == 'AB' or args.images == 'B':
    if args.imga_path is not None:
        imgA = Image.open(os.path.join(args.imga_path)).convert("RGB")
        imgA.load()
        imgA = transform(imgA)
    else:
        imgA, _ = next(iter(dataloader))
    fakeB = genB(imgA)

if args.images == 'AB' or args.images == 'A':
    if args.imgb_path is not None:
        imgB = Image.open(os.path.join(args.imgb_path)).convert("RGB")
        imgB.load()
        imgB = transform(imgB)
    else:
        _, imgB = next(iter(dataloader))
    fakeA = genA(imgB)

if args.images == 'AB' or args.images == 'A' and fakeA is not None:
    imgB = 0.5 * (imgB.data + 1.0)
    fakeA = 0.5 * (fakeA.data + 1.0)
    utils.save_image(fakeA.detach(),
                     f"/test_results/realB.png",
                     normalize=True)
    utils.save_image(fakeA.detach(),
                     f"/test_results/fakeA.png",
                     normalize=True)

if args.images == 'AB' or args.images == 'B' and fakeB is not None:
    imgA = 0.5 * (imgA.data + 1.0)
    utils.save_image(fakeA.detach(),
                     f"/test_results/realA.png",
                     normalize=True)
    fakeB = 0.5 * (fakeB.data + 1.0)
    utils.save_image(fakeB.detach(),
                     f"/test_results/results/fakeB.png",
                     normalize=True)
