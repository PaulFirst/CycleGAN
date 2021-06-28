from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob


class CustomDataset(Dataset):
    def __init__(self, directory, mode):
        super(CustomDataset, self).__init__()
        self.mode = mode
        self.directory = directory
        self.pathA = ''
        self.pathB = ''
        if self.mode == 'train':
            self.pathA = 'trainA'
            self.pathB = 'trainB'
        elif self.mode == 'test':
            self.pathA = 'testA'
            self.pathB = 'testB'
        self.trainA_path = os.listdir(os.path.join(self.directory, self.pathA))
        self.trainB_path = os.listdir(os.path.join(self.directory, self.pathB))
        print(len(self.trainA_path))
        self.len = min(len(self.trainA_path), len(self.trainB_path))

    def load_samples(self, index):
        imgA = Image.open(os.path.join(self.directory, self.pathA, self.trainA_path[index])).convert("RGB")
        imgB = Image.open(os.path.join(self.directory, self.pathB, self.trainB_path[index])).convert("RGB")
        imgA.load()
        imgB.load()
        return imgA, imgB

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        transform = ''
        if self.mode == 'train':
            transform = tf.Compose([tf.ToTensor(), tf.RandomHorizontalFlip(),
                                    tf.Resize((286, 286)), tf.RandomCrop((256, 256)),
                                    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif self.mode == 'test':
            transform = tf.Compose(
                [tf.ToTensor(), tf.Resize((256, 256)), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        imgA, imgB = self.load_samples(index)
        imgA = transform(imgA)
        imgB = transform(imgB)

        return imgA, imgB


if __name__ == '__main__':
    dataset = CustomDataset('D:\\Projects\\CycleGAN\\horse2zebra', 'train')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=3)
    imgsA, imgsB = next(iter(dataloader))
    print(imgsA.shape)

    fig, axes = plt.subplots(3, 2)
    for i in range(3):
        axes[i, 0].imshow(imgsA[i].detach().to('cpu').numpy().transpose(1, 2, 0))
        axes[i, 1].imshow(imgsB[i].detach().to('cpu').numpy().transpose(1, 2, 0))
    plt.show()
