import os
import pandas as pd
from PIL import Image

# Define the folder path and CSV filename
folder_path = 'futhark_letters/'
csv_filename = 'futhark_letters_images.csv'

# Initialize lists to store image file names and labels
image_files = []
labels = []

# Iterate through the folder and collect image file names and labels
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Resize the image to 48x48
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        img = img.resize((256, 256))  # Changed to 256x256
        img.save(img_path)

        image_files.append(img_path)
        labels.append('cyberpunk')  # Assign a default label

# Create a Pandas DataFrame
df = pd.DataFrame({
    'file_path': image_files,
    'rune': labels  # Use 'rune' as the column name to match your existing dataset
})

# Save the DataFrame to a CSV file
df.to_csv(csv_filename, index=False)