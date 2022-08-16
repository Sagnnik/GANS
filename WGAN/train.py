import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator, initialize_weights

#Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMG_SIZE = 64
Z_DIM = 128
FEATURES_CRIT = 64
FEATURES_GEN = 64
WEIGHT_CLIP = 0.01
CHANNELS_IMG = 3
NUMBER_EPOCHS = 10
CRIT_ITR = 5

transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])])

#dataset = datasets.MNIST(root="/WGAN/dataset", train=True, transform=transform, download=True)
#dataset = datasets.ImageFolder("WGAN\images", transform=transform)
dataset = datasets.CelebA(root="/WGAN/CelebA/", split="train", transform=transform, download=True)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRIT).to(device)
initialize_weights(gen)
initialize_weights(critic)

optim_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
optim_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

fized_noise = torch.randn(size=(BATCH_SIZE, CHANNELS_IMG, 1, 1)).to(device)

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"WGAN/logs/real")
writer_fake = SummaryWriter(f"WGAN/logs/fake")
step=0

gen.train()
critic.train()

for epoch in range(NUMBER_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        #Train Critic: max E(critic(real)) - E(critic(fake))
        for itr in range(CRIT_ITR):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            critic_loss = -(torch.mean(critic_real)- torch.mean(critic_fake))
            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optim_critic.step()

            #Clipping the weights between -0.01 and 0.01
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        #Train the Generator: max E(critic(gen(fake))) <--> min -E(critic(gen(fake)))
        gen_fake = gen(fake).reshape(-1)
        gen_loss = -(torch.mean(critic(gen_fake)))
        gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{NUMBER_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {critic_loss:.4f}, loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
            critic.train()


