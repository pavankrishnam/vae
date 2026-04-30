import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("vae_images", exist_ok=True)

transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.fc2 = nn.Linear(20, 400)
        self.fc3 = nn.Linear(400, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std*torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.fc3(torch.relu(self.fc2(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_fn(recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    KL = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KL

model = VAE().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def save_images(epoch):
    x, _ = next(iter(loader))
    x = x.view(-1,784).to(device)
    recon,_,_ = model(x)
    recon = recon.view(-1,1,28,28).cpu()
    grid = torchvision.utils.make_grid(recon[:16], nrow=4)
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"vae_images/epoch_{epoch}.png")
    plt.close()

for epoch in range(5):
    for x,_ in loader:
        x = x.view(-1,784).to(device)
        recon, mu, logvar = model(x)
        loss = loss_fn(recon, x, mu, logvar)

        opt.zero_grad()
        loss.backward()
        opt.step()

    save_images(epoch)
    print(f"[VAE] Epoch {epoch} | Loss: {loss.item():.0f}")
