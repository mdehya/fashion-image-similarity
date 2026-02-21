import streamlit as st
import numpy as np
import torch
import os
import pandas as pd
from PIL import Image

from src.model import get_model, get_transform
from src.similarity import compute_similarity


# LOAD DATA
embeddings = np.load("embeddings/image_embeddings.npy")
image_ids = np.load("embeddings/image_ids.npy")

image_folder = "data/selected_images"


# LOAD METADATA
df = pd.read_csv("data/styles.csv", engine="python", on_bad_lines="skip")
id_to_category = dict(zip(df["id"], df["articleType"]))

# ambil kategori yang benar-benar dipakai
used_categories = []
for img_name in image_ids:
    img_id = int(img_name.split(".")[0])
    cat = id_to_category.get(img_id)
    if cat:
        used_categories.append(cat)

categories = sorted(list(set(used_categories)))


# MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device)
transform = get_transform()


# FUNCTION
def get_embedding(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy().flatten()


def search_similar(query_embedding, top_k=5, selected_category="All"):
    similarities = compute_similarity(query_embedding, embeddings)

    indices = list(range(len(similarities)))

    if selected_category != "All":
        filtered = []
        for i in indices:
            img_id = int(image_ids[i].split(".")[0])
            if id_to_category.get(img_id) == selected_category:
                filtered.append(i)
        indices = filtered

    ranked = sorted(indices, key=lambda x: similarities[x], reverse=True)
    return ranked[:top_k], similarities



# UI
st.set_page_config(page_title="Fashion Similarity", layout="wide")

st.title("👕 Fashion Image Similarity Search")
st.markdown("Upload gambar dan temukan item fashion yang mirip secara visual.")

# sidebar
st.sidebar.header("⚙️ Pengaturan")

top_k = st.sidebar.slider("Jumlah hasil (Top-K)", 1, 10, 5)

selected_category = st.sidebar.selectbox(
    "Filter kategori",
    ["All"] + categories
)

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png"])


# MAIN
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📌 Query Image")
    st.image(image, width=250)

    # embedding
    query_embedding = get_embedding(image)

    # search
    top_indices, similarities = search_similar(
        query_embedding,
        top_k=top_k,
        selected_category=selected_category
    )

    st.subheader("🔍 Similar Images")

    cols = st.columns(top_k)

    for i, idx in enumerate(top_indices):
        img_path = os.path.join(image_folder, image_ids[idx])
        img = Image.open(img_path)

        img_id = int(image_ids[idx].split(".")[0])
        category = id_to_category.get(img_id, "Unknown")

        cols[i].image(
            img,
            caption=f"{category}\nScore: {similarities[idx]:.2f}"
        )