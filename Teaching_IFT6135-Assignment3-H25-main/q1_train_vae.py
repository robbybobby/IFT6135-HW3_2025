from __future__ import print_function
import argparse
from ast import mod
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from q1_vae import *
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.epochs = 20

gdrive_dir = '../IFT6135-Assignment3/output'
save_dir = os.path.join(gdrive_dir, 'vae')
os.makedirs(gdrive_dir, exist_ok=True)

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(f'{save_dir}/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(f'{save_dir}/data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    ## TO DO: Implement the loss function using your functions from q1_solution.py
    ## use the following structure:
    # kl = kl_gaussian_gaussian_analytic(mu_q=?, logvar_q=?, mu_p=?, logvar_p=?).sum()
    # recon_loss = (?).sum()
    # return recon_loss + kl
    mu_p = torch.zeros_like(mu)
    logvar_p = torch.zeros_like(logvar)
    kl = kl_gaussian_gaussian_analytic(mu_q=mu, logvar_q=logvar, mu_p=mu_p, logvar_p=logvar_p).sum()
    recon_loss = -(log_likelihood_bernoulli(recon_x, x.view(-1, 784))).sum()
    return recon_loss + kl

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_t_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average training loss: {:.4f}'.format(epoch, avg_t_loss))
    return avg_t_loss

def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()
    avg_v_loss = val_loss / len(test_loader.dataset)
    print('====> Epoch: {} Average validation loss: {:.4f}'.format(epoch, avg_v_loss))
    return avg_v_loss

def plot_losses(train_loss, valid_loss, save_dir = save_dir):
    epochs = list(range(1, len(train_loss)+1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, color='blue', label='Training Loss')
    plt.plot(epochs, valid_loss, color='green', label='Validation Loss')
    plt.axhline(y=104, color='red', linestyle='--', label='Target Threshold (104)')

    plt.xlabel('EPOCHS')
    plt.ylabel('LOSSES')
    plt.title('VAE Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{save_dir}/vae_TVL_plot.png")
    plt.show()

def generate_samples(model, nsmpl=16, latdim=20):
    model.eval()
    with torch.no_grad():
        z = torch.randn(nsmpl, latdim).to(device)
        samples = model.decode(z).cpu()
        return samples.view(nsmpl, 1, 28, 28)

def show_samples(samples, nrow=4, title="Generated Samples", save_dir=save_dir):
    n_samples = samples.size(0)
    fig, axs = plt.subplots(nrow, nrow, figsize=(6, 6))
    axs = axs.flatten()
    for i in range(n_samples):
        axs[i].imshow(samples[i].squeeze(), cmap='gray')
        axs[i].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_samples.png")
    plt.show()

def latent_traversal_grid(model, latent_dim=20, steps=5, epsilon=2.0):
    model.eval()
    with torch.no_grad():
        base_z = torch.randn(1, latent_dim).to(device)
        traversal_range = torch.linspace(-epsilon, epsilon, steps).to(device)
        images = []
        for i in range(latent_dim):
            row = []
            for shift in traversal_range:
                z_new = base_z.clone()
                z_new[0, i] += shift
                img = model.decode(z_new).view(28, 28).cpu()
                row.append(img)
            images.append(row)
        return images

def show_latent_traversals(images, steps=5, save_dir=save_dir):
    latent_dim = len(images)
    fig, axs = plt.subplots(latent_dim, steps, figsize=(steps, latent_dim))
    for i in range(latent_dim):
        for j in range(steps):
            axs[i, j].imshow(images[i][j], cmap='gray')
            axs[i, j].axis('off')
    plt.suptitle("VAE Latent Traversals")
    plt.tight_layout(pad = 2)
    plt.savefig(f"{save_dir}/vae_latent_traversals.png")
    plt.show()

def interpolate_latent_vs_data(model, latent_dim=20, steps=11):
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(1, latent_dim).to(device)
        z1 = torch.randn(1, latent_dim).to(device)
        x0 = model.decode(z0).view(28, 28).to(device)
        x1 = model.decode(z1).view(28, 28).to(device)
        alphas = torch.linspace(0, 1, steps).to(device)
        latent_imgs = []
        for alpha in alphas:
            z_alpha = alpha * z0 + (1 - alpha) * z1
            x_alpha = model.decode(z_alpha).view(28, 28).cpu()
            latent_imgs.append(x_alpha)
        data_imgs = []
        for alpha in alphas:
            x_hat = alpha * x0 + (1 - alpha) * x1
            data_imgs.append(x_hat)
        return latent_imgs, data_imgs

def plot_interpolations(latent_imgs, data_imgs, save_dir=save_dir):
    steps = len(latent_imgs)
    fig, axs = plt.subplots(2, steps, figsize=(steps, 2))
    for i in range(steps):
        axs[0, i].imshow(latent_imgs[i].cpu().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(data_imgs[i].cpu().numpy(), cmap='gray')
        axs[1, i].axis('off')
    axs[0, 0].set_ylabel("Latent", fontsize=12)
    axs[1, 0].set_ylabel("Data", fontsize=12)
    plt.suptitle("Interpolation in Latent vs Data Space")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_latent_data_space_plot.png")
    plt.show()
        
if __name__ == "__main__":
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        train_losses.append(train(epoch))
        val_losses.append(validate(epoch))

    torch.save(model.state_dict(), 'vae_model.pt')

    plot_losses(train_losses, val_losses)

    show_samples(generate_samples(model, nsmpl=16))

    show_latent_traversals(latent_traversal_grid(model, latent_dim=20, steps=5, epsilon=2.5))

    latentimgs, dataimgs = interpolate_latent_vs_data(model, latent_dim=20, steps=11)
    plot_interpolations(latentimgs, dataimgs)