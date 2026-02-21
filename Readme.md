# Fashion Image Similarity Search

Project ini bertujuan untuk membangun sistem pencarian gambar produk fashion berdasarkan kemiripan visual menggunakan Deep Learning.



## Overview

Sistem ini memungkinkan pengguna untuk:
- Mengunggah gambar produk fashion (query image)
- Mencari gambar lain yang paling mirip secara visual
- Menampilkan Top-K hasil kemiripan

Pendekatan yang digunakan:
- Feature Extraction dengan ResNet-50 (pretrained ImageNet)
- Representasi gambar dalam bentuk embedding vector
- Perhitungan kemiripan menggunakan Cosine Similarity



## Methodology

Alur sistem:

1. Preprocessing gambar
   - Resize ke 224x224
   - Normalisasi pixel

2. Feature Extraction
   - Menggunakan ResNet-50 (tanpa layer klasifikasi)
   - Output: embedding vector (2048 dimensi)

3. Similarity Computation
   - Menggunakan cosine similarity

4. Retrieval
   - Mengambil Top-K gambar paling mirip



## Dataset

Dataset yang digunakan:
- Fashion Product Images Dataset (Kaggle)

Link dataset:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

### Cara penggunaan dataset:
1. Download dataset dari Kaggle
2. Ekstrak ke folder:



## Installation

```bash
git clone https://github.com/mdehya/fashion-image-similarity.git
cd fashion-image-similarity

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

Usage
1. Generate Embedding 

python extract_embeddings.py

Output: 
- embeddings/image_embeddings.npy
- embeddings/image_ids.npy

2. Run App
streamlit run app.py

