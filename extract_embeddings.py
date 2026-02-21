import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.model import get_model, get_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(device)
transform = get_transform()

image_folder = "data/selected_images"
image_files = os.listdir(image_folder)

embeddings = []
image_ids = []

for img_name in tqdm(image_files):
    img_path = os.path.join(image_folder, img_name)
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(img_tensor)
    
    embedding = embedding.squeeze().cpu().numpy()  # jadi 2048 vector
    
    embeddings.append(embedding)
    image_ids.append(img_name)

embeddings = np.array(embeddings)

# Simpan
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/image_embeddings.npy", embeddings)
np.save("embeddings/image_ids.npy", image_ids)

print("Embedding extraction selesai.")
print("Shape:", embeddings.shape)
