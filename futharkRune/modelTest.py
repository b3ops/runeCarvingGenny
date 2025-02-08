import torch
from torchvision import transforms
from torchvision.utils import save_image
from runeCarving import Generator  # Replace 'your_model_file' with the actual file where Generator is defined
from PIL import Image

# Ask user for device preference
use_gpu = input("Do you want to use GPU? (y/n): ").lower() == 'y'

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")

    print("Using GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Load the model
checkpoint = torch.load('final_model.pth', map_location=device, weights_only=True)

# Now, you can debug or perform any setup before loading the model state
print("Checkpoint loaded. Ready to instantiate models.")

G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)

# Load state dicts
G_XtoY.load_state_dict(checkpoint['G_XtoY_state_dict'])
G_YtoX.load_state_dict(checkpoint['G_YtoX_state_dict'])

# # Assuming you saved the entire model object
# G_XtoY = torch.load('final_model.pth')['G_XtoY'].to(device)
# G_YtoX = torch.load('final_model.pth')['G_YtoX'].to(device)

# Set to evaluation mode
G_XtoY.eval()
G_YtoX.eval()

print("Models instantiated and loaded. Ready for use.")

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load an image (replace 'path_to_image' with the actual path)
original_image = Image.open('img/image.jpg').convert('RGB')
input_image = transform(original_image).unsqueeze(0).to(device)  # Add batch dimension

with torch.no_grad():
    # Translate from X to Y (wood carving to futhark letter or vice versa)
    translated_image = G_XtoY(input_image)  # Use G_YtoX for the reverse translation
    # The model outputs tensor, so we need to convert it back to an image
    output_image = translated_image.squeeze(0).cpu()
    output_image = output_image * 0.5 + 0.5  # Denormalize

# Save the image
save_image(output_image, 'translated_image.png')

# Or you can display it directly
import matplotlib.pyplot as plt

plt.imshow(output_image.permute(1, 2, 0).numpy())
plt.axis('off')
plt.show()
