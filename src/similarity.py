import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query_embedding, embeddings):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]
    return similarities


def get_top_k(similarities, top_k=5):
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices