import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os
import logging

logging.basicConfig(level=logging.INFO)

class WoodCarvingDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        if not self.images:
            logging.error(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        try:
            image = Image.open(img_name)
            image = self.transform(image)
            return image
        except Exception as e:
            logging.error(f"Error loading image {img_name}: {str(e)}")
            return None

class FutharkLetterDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        if not self.images:
            logging.error(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        try:
            image = Image.open(img_name)
            image = image.convert('RGB')
            image = self.transform(image)
            futhark_letter = self.images[idx].split('-')[0]
            return image, futhark_letter
        except Exception as e:
            logging.error(f"Error loading image {img_name}: {str(e)}")
            return None, None

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

wood_carving_dataset = WoodCarvingDataset('wood_carvings', transform)
futhark_letter_dataset = FutharkLetterDataset('futhark_letters', transform)

# Create your DataLoaders
wood_carving_loader = torch.utils.data.DataLoader(wood_carving_dataset, batch_size=4, shuffle=True)
futhark_letter_loader = torch.utils.data.DataLoader(futhark_letter_dataset, batch_size=4, shuffle=True)

G_XtoY = Generator()
G_YtoX = Generator()
D_X = Discriminator()
D_Y = Discriminator()

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G = torch.optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=0.001)
optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=0.001)
optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=0.001)

def train_model(G_XtoY, G_YtoX, D_X, D_Y, wood_carving_loader, futhark_letter_loader, optimizer_G, optimizer_D_X, optimizer_D_Y, criterion_GAN, criterion_cycle, criterion_identity, num_epochs):
    try:
        for epoch in range(num_epochs):
            for i, (wood_carving_batch, futhark_letter_batch) in enumerate(zip(wood_carving_loader, futhark_letter_loader)):
                wood_carving_images = wood_carving_batch[0]
                futhark_letter_images, futhark_letters = futhark_letter_batch

                if wood_carving_images is None or futhark_letter_images is None:
                    continue
                
                optimizer_G.zero_grad()
                fake_futhark_letter_images = G_XtoY(wood_carving_images)
                fake_wood_carving_images = G_YtoX(futhark_letter_images)

                from torchvision.transforms.functional import resize
                resized_fake_futhark = resize(fake_futhark_letter_images, (256, 256))
                resized_fake_wood = resize(fake_wood_carving_images, (256, 256))

                cycle_loss = criterion_cycle(wood_carving_images, G_YtoX(resized_fake_futhark)) + criterion_cycle(futhark_letter_images, G_XtoY(resized_fake_wood))
                identity_loss = criterion_identity(wood_carving_images, G_XtoY(wood_carving_images)) + criterion_identity(futhark_letter_images, G_YtoX(futhark_letter_images))
                GAN_loss = criterion_GAN(D_Y(resized_fake_futhark), torch.ones_like(D_Y(resized_fake_futhark))) + criterion_GAN(D_X(resized_fake_wood), torch.ones_like(D_X(resized_fake_wood)))
                total_loss = cycle_loss + identity_loss + GAN_loss
                total_loss.backward()
                optimizer_G.step()

                optimizer_D_X.zero_grad()
                optimizer_D_Y.zero_grad()
                D_X_loss = criterion_GAN(D_X(wood_carving_images), torch.ones_like(D_X(wood_carving_images))) + criterion_GAN(D_X(resized_fake_wood.detach()), torch.zeros_like(D_X(resized_fake_wood.detach())))
                D_Y_loss = criterion_GAN(D_Y(futhark_letter_images), torch.ones_like(D_Y(futhark_letter_images))) + criterion_GAN(D_Y(resized_fake_futhark.detach()), torch.zeros_like(D_Y(resized_fake_futhark.detach())))
                D_X_loss.backward()
                D_Y_loss.backward()
                optimizer_D_X.step()
                optimizer_D_Y.step()

                print(f"Epoch {epoch+1}, Iteration {i+1}", flush=True)
                print(f"Cycle Loss: {cycle_loss.item():.4f}", flush=True)
                print(f"Identity Loss: {identity_loss.item():.4f}", flush=True)
                print(f"GAN Loss: {GAN_loss.item():.4f}", flush=True)
                print(f"Total Loss: {total_loss.item():.4f}", flush=True)
                print(f"D_X Loss: {D_X_loss.item():.4f}", flush=True)
                print(f"D_Y Loss: {D_Y_loss.item():.4f}", flush=True)
                print("-" * 50, flush=True)

                # Save model checkpoints
                if (epoch + 1) % 5 == 0 and (i + 1) % 10 == 0:  # Example: Save every 5 epochs and every 10 iterations within an epoch
                    torch.save({
                        'epoch': epoch + 1,
                        'iteration': i + 1,
                        'G_XtoY_state_dict': G_XtoY.state_dict(),
                        'G_YtoX_state_dict': G_YtoX.state_dict(),
                        'D_X_state_dict': D_X.state_dict(),
                        'D_Y_state_dict': D_Y.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_X_state_dict': optimizer_D_X.state_dict(),
                        'optimizer_D_Y_state_dict': optimizer_D_Y.state_dict(),
                        'loss': total_loss.item(),
                    }, f"checkpoint_epoch{epoch+1}_iter{i+1}.pth")
                    print(f"Checkpoint saved at epoch {epoch+1}, iteration {i+1}", flush=True)

                print("Wood carving images shape:", wood_carving_images.shape)
                print("Fake futhark letter images shape:", fake_futhark_letter_images.shape)
                print("Fake wood carving images shape:", fake_wood_carving_images.shape)
                print("Futhark letter images shape:", futhark_letter_images.shape)
                print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
# Then calculate loss
num_epochs = 100
train_model(G_XtoY, G_YtoX, D_X, D_Y, wood_carving_loader, futhark_letter_loader, optimizer_G, optimizer_D_X, optimizer_D_Y, criterion_GAN, criterion_cycle, criterion_identity, num_epochs)