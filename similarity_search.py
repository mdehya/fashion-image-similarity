import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding
embeddings = np.load("embeddings/image_embeddings.npy")
image_ids = np.load("embeddings/image_ids.npy")

image_folder = "data/selected_images"


# PILIH QUERY IMAGE

query_index = 100  # ganti index untuk coba gambar lain

query_embedding = embeddings[query_index].reshape(1, -1)


# HITUNG COSINE SIMILARITY

similarities = cosine_similarity(query_embedding, embeddings)[0]


# AMBIL TOP-K

top_k = 6  # termasuk dirinya sendiri

top_indices = similarities.argsort()[-top_k:][::-1]


# TAMPILKAN HASIL

plt.figure(figsize=(12, 4))

for i, idx in enumerate(top_indices):
    img_path = os.path.join(image_folder, image_ids[idx])
    img = Image.open(img_path)
    
    plt.subplot(1, top_k, i+1)
    plt.imshow(img)
    plt.title(f"{similarities[idx]:.2f}")
    plt.axis("off")

plt.suptitle("Query dan Similar Images")
plt.show()