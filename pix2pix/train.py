import torch
import torchvision
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataloader import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

def train_fn(gen, disc, loader, opt_disc, opt_gen, l1_loss, bce_loss, g_scalar, d_scalar):
    loop=tqdm(loader, leave=True)
    
    for idx, (x,y) in enumerate(loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        #Training of Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_fake_loss + D_real_loss)/2
        
        disc.zero_grad()
        d_scalar.scale(D_loss).backward()
        d_scalar.step(opt_disc)
        d_scalar.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scalar.scale(G_loss).backward()
        g_scalar.step(opt_gen)
        g_scalar.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

        
       


def main():
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.99))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5,0.99))

    BCE_loss = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, lr=config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, lr=config.LEARNING_RATE)

    train_dataset = MapDataset("pix2pix/dataset/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scalar = torch.cuda.amp.GradScaler() #float16 training
    d_scalar = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset("pix2pix/dataset/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    gen.train()
    disc.train()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_loss, BCE_loss, g_scalar, d_scalar)

        if config.SAVE_MODEL and epoch % 5==0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="dataset/evaluation")



if __name__ == "__main__":
    main()