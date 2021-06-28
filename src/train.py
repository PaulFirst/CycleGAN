import argparse
from src.model.generator import Generator
from src.model.discriminator import Discriminator
import torch
from src.data.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from src.utils.buffer import ImagePool
from tqdm import tqdm
import torchvision.utils as utils
import matplotlib.pyplot as plt
import itertools
from src.utils.utils import denormalize, weights_init
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# EPOCHS = 100
# LR = 0.0002
# LAMBDA_IDENT = 0.5
# LAMBDA_A = 10
# LAMBDA_B = 10
# BUFFER_SIZE = 50

parser = argparse.ArgumentParser(description='CycleGAN_Train')
parser.add_argument('--dataset_dir', type=str, help='директория набора данных для обучения')
parser.add_argument('--batch_size', type=int, default=1, help='размер пакета обучения')
parser.add_argument('--total_epochs', type=int, default=50, help='общее число эпох обучения')
parser.add_argument('--buffer_size', type=int, default=50, help='размер буффера для дискриминатора')
parser.add_argument('--lr', type=float, default=0.0002, help='шаг обучения')
parser.add_argument('--lr_beta', type=float, default=0.5, help='величина beta для Adam оптимизатора')
parser.add_argument('--train_pics', type=bool, default=False, help='отображение промежуточных резульатов')
parser.add_argument('--weight_path', type=str, default=None, help='путь до весов модели')
parser.add_argument('--view_freq', type=int, default=100, help='частота просмотра результатов обучения')
parser.add_argument('--lamda_a', type=float, default=10.0)
parser.add_argument('--lamda_b', type=float, default=10.0)
parser.add_argument('--lamda_id', type=float, default=0.5)

args = parser.parse_args()

# датасет
dataset = CustomDataset(args.dataset_dir, 'train')
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
fakeABuffer = ImagePool(args.buffer_size)
fakeBBuffer = ImagePool(args.buffer_size)

# сеть
genA = Generator(3, 3).to(DEVICE)
genA.apply(weights_init)
genB = Generator(3, 3).to(DEVICE)
genB.apply(weights_init)
discA = Discriminator(3).to(DEVICE)
discA.apply(weights_init)
discB = Discriminator(3).to(DEVICE)
discB.apply(weights_init)
# загрузка сети
current_epoch = 0
if args.weight_path is not None:
    checkpoint = torch.load(args.weight_path)
    genA.load_state_dict(checkpoint['genA_state_dict'])
    genB.load_state_dict(checkpoint['genB_state_dict'])
    discA.load_state_dict(checkpoint['discA_state_dict'])
    discB.load_state_dict(checkpoint['discB_state_dict'])
    current_epoch = checkpoint['epoch']

# функции потерь
identCriterion = nn.L1Loss()
adversarialCriterion = nn.MSELoss()
cycleCriterion = nn.L1Loss()

# оптимизаторы
optimG = torch.optim.Adam(itertools.chain(genA.parameters(), genB.parameters()),
                          lr=args.lr, betas=(args.lr_beta, 0.999))
optimDiscA = Adam(discA.parameters(), args.lr, betas=(args.lr_beta, 0.999))
optimDiscB = Adam(discB.parameters(), args.lr, betas=(args.lr_beta, 0.999))

# цикл обучения
log_template = "\nEpoch {ep:03d} G_loss: {g_loss:0.4f} \
    D_A_loss {da_loss:0.4f} D_B_loss {db_loss:0.4f}"

os.makedirs('./results/resultsB/fake', exist_ok=True)
os.makedirs('./results/resultsB/real', exist_ok=True)
os.makedirs('./results/resultsB/reconstructed', exist_ok=True)
os.makedirs('./results/resultsA/fake', exist_ok=True)
os.makedirs('./results/resultsA/real', exist_ok=True)
os.makedirs('./results/resultsA/reconstructed', exist_ok=True)

with tqdm(desc="epoch", total=args.total_epochs) as pbar:
    for epoch in range(current_epoch, args.total_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        generator_loss = []
        discA_loss = []
        discB_loss = []
        for i, (realA, realB) in progress_bar:
            realA = realA.to(DEVICE)
            realB = realB.to(DEVICE)
            # genA
            optimG.zero_grad()

            fakeA = genA(realB)
            fakeB = genB(realA)
            # discriminator
            rateFakeA = discA(fakeA)
            rateFakeB = discB(fakeB)
            # cycle
            reconstructB = genB(fakeA)
            reconstructA = genA(fakeB)

            # ident
            sameA = genA(realA)
            sameB = genB(realB)

            # ident losses
            identLossA = identCriterion(sameA, realA) * args.lamda_a * args.lambda_id
            identLossB = identCriterion(sameB, realB) * args.lamda_b * args.lambda_id

            # adversarial losses
            genBRealLabels = torch.ones(rateFakeB.shape, device=DEVICE)
            advLossB = adversarialCriterion(rateFakeB, genBRealLabels)
            genARealLabels = torch.ones(rateFakeA.shape, device=DEVICE)
            advLossA = adversarialCriterion(rateFakeA, genARealLabels)

            BCycleLoss = cycleCriterion(reconstructB, realB) * args.lamda_b
            ACycleLoss = cycleCriterion(reconstructA, realA) * args.lamda_a

            genLoss = identLossA + identLossB + advLossA + advLossB + ACycleLoss + BCycleLoss
            genLoss.backward()
            optimG.step()

            # discA
            optimDiscA.zero_grad()

            fakeAD = fakeABuffer.query(fakeA)
            predsRealA = discA(realA)
            discARealLabels = torch.ones(predsRealA.shape, device=DEVICE)
            realDiscALoss = adversarialCriterion(predsRealA, discARealLabels)
            predsFakeA = discA(fakeAD.detach())
            discAFakeLabels = torch.zeros(predsFakeA.shape, device=DEVICE)
            fakeDiscALoss = adversarialCriterion(predsFakeA, discAFakeLabels)
            discALoss = (realDiscALoss + fakeDiscALoss) / 2
            discA_loss.append(discALoss)
            discALoss.backward()
            optimDiscA.step()

            # discB
            optimDiscB.zero_grad()
            fakeBD = fakeBBuffer.query(fakeB)
            predsRealB = discB(realB)
            discBRealLabels = torch.ones(predsRealB.shape, device=DEVICE)
            realDiscBLoss = adversarialCriterion(predsRealB, discBRealLabels)
            predsFakeB = discB(fakeBD.detach())
            discBFakeLabels = torch.zeros(predsFakeB.shape, device=DEVICE)
            fakeDiscBLoss = adversarialCriterion(predsFakeB, discBFakeLabels)
            discBLoss = (realDiscBLoss + fakeDiscBLoss) / 2
            discB_loss.append(discBLoss)
            discBLoss.backward()
            optimDiscB.step()

            progress_bar.set_description(
                f"[{epoch}/{args.total_epochs - 1}][{i}/{len(dataloader) - 1}] "
                f"Loss_G: {genLoss.item():.4f} "
                f"Loss_D: {(discALoss + discBLoss).item():.4f} "
                f"loss_G_adv: {(advLossB + advLossB).item():.4f} "
                f"Loss_G_ident: {(identLossA + identLossB).item():.4f} "
                f"loss_G_cycle: {(ACycleLoss + BCycleLoss).item():.4f}")

            if i % args.view_freq == 0:
                if args.train_pics:
                    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
                    axes[0, 0].imshow(realA[0].detach().to('cpu').numpy().transpose(1, 2, 0))
                    axes[0, 0].set_title('realA')
                    axes[0, 1].imshow(realB[0].detach().to('cpu').numpy().transpose(1, 2, 0))
                    axes[0, 1].set_title('realB')
                    axes[1, 0].imshow(denormalize(fakeB[0].detach().to('cpu').numpy()).transpose(1, 2, 0))
                    axes[1, 0].set_title('fakeA')
                    axes[1, 1].imshow(denormalize(fakeA[0].detach().to('cpu').numpy()).transpose(1, 2, 0))
                    axes[1, 1].set_title('fakeB')
                    axes[2, 0].imshow(denormalize(reconstructA[0].detach().to('cpu').numpy()).transpose(1, 2, 0))
                    axes[2, 0].set_title('reconstructA')
                    axes[2, 1].imshow(denormalize(reconstructB[0].detach().to('cpu').numpy()).transpose(1, 2, 0))
                    axes[2, 1].set_title('reconstructB')
                    plt.show()

                # денормализация и сохранение
                fakeA = genA(realB)
                fakeB = genB(realA)
                reconstructA = genA(fakeB)
                reconstructB = genA(fakeA)

                fakeA = 0.5 * (fakeA.data + 1.0)
                fakeB = 0.5 * (fakeB.data + 1.0)

                realA = 0.5 * (realA.data + 1.0)
                realB = 0.5 * (realB.data + 1.0)

                reconstructA = 0.5 * (reconstructA.data + 1.0)
                reconstructB = 0.5 * (reconstructB.data + 1.0)

                utils.save_image(fakeA.detach(),
                                 f"./results/resultsB/fake/fake_epoch_{epoch}.png",
                                 normalize=True)
                utils.save_image(fakeB.detach(),
                                 f"./results/resultsA/fake/fake_epoch_{epoch}.png",
                                 normalize=True)
                utils.save_image(realA.detach(),
                                 f"./results/resultsA/real/real_epoch_{epoch}.png",
                                 normalize=True)
                utils.save_image(realB.detach(),
                                 f"./results/resultsB/real/real_epoch_{epoch}.png",
                                 normalize=True)
                utils.save_image(reconstructA.detach(),
                                 f"./results/resultsA/reconstructed/recostruct_epoch_{epoch}.png",
                                 normalize=True)
                utils.save_image(reconstructB.detach(),
                                 f"./results/resultsB/reconstructed/recostruct_epoch_{epoch}.png",
                                 normalize=True)

        pbar.update(1)
        torch.save({'genA_state_dict': genA.state_dict(),
                    'genB_state_dict': genB.state_dict(),
                    'discA_state_dict': discA.state_dict(),
                    'discB_state_dict': discB.state_dict(),
                    'epoch': epoch + 1}, args.weight_path)
        generator_loss = sum(generator_loss) / (len(generator_loss) + 1e-8)
        discA_loss = sum(discA_loss) / (len(discA_loss) + 1e-8)
        discB_loss = sum(discB_loss) / (len(discB_loss) + + 1e-8)
        tqdm.write(log_template.format(ep=epoch + 1, g_loss=generator_loss, da_loss=discA_loss, db_loss=discB_loss))
