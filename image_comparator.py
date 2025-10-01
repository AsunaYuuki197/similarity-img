import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'reranking')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'hp_and_cs')))

import numpy as np
from numpy.linalg import norm
from PIL import Image
import streamlit as st
from config import MATRIX_THRESHOLD, FEATURE_THRESHOLD, DEVICE
from get_embedding import get_image_embedding
from reranking.reranker.qe_reranking import aqe_reranking
from reranking.reranker.cas_reranking import cas_reranking

def compare_images_matrix(img1, img2):
    """
    Compare two images using pixel-wise mean absolute difference.
    Args:
        img1, img2 (PIL.Image): Images to compare.
    Returns:
        float: Difference score (lower is more similar).
    """
    if img1 is None or img2 is None:
        return float('inf')
    
    try:
        img1_array = np.array(img1.resize((224, 224)))
        img2_array = np.array(img2.resize((224, 224)))

        if len(img1_array.shape) > 2 and img1_array.shape[2] == 4:  # RGBA to RGB
            img1_array = img1_array[:, :, :3]
        if len(img2_array.shape) > 2 and img2_array.shape[2] == 4:  # RGBA to RGB
            img2_array = img2_array[:, :, :3]

        if len(img1_array.shape) == 2:  # Grayscale to RGB
            img1_array = np.stack([img1_array] * 3, axis=-1)
        if len(img2_array.shape) == 2:  # Grayscale to RGB
            img2_array = np.stack([img2_array] * 3, axis=-1)

        img1_array = img1_array / 255.0
        img2_array = img2_array / 255.0

        return np.mean(np.abs(img1_array - img2_array))
    except Exception as e:
        st.error(f"Error in matrix comparison: {e}")
        return float('inf')

def compare_images_feature(img1, img2):
    """
    Compare two images using cosine similarity of their embeddings.
    Args:
        img1, img2 (PIL.Image): Images to compare.
    Returns:
        float: Cosine similarity score (higher is more similar).
    """
    if img1 is None or img2 is None:
        return 0.0
    
    try:
        img1_embedding = get_image_embedding(img1)
        img2_embedding = get_image_embedding(img2)
        return np.dot(img1_embedding, img2_embedding) / (norm(img1_embedding) * norm(img2_embedding))
    except Exception as e:
        st.error(f"Error in feature comparison: {e}")
        return 0.0

def compare_images_cosine(query_embedding, db_embeddings):
    """
    Compare query image to database images using cosine similarity.
    Args:
        query_embedding (np.ndarray): Query image embedding.
        db_embeddings (np.ndarray): Database image embeddings.
    Returns:
        list: Ranked indices (highest similarity first).
    """
    try:
        scores = np.dot(query_embedding, db_embeddings.T)
        return np.argsort(-scores).tolist()
    except Exception as e:
        st.error(f"Error in cosine similarity comparison: {e}")
        return []

def compare_images_aqe(query_embedding, db_embeddings):
    """
    Compare query image to database images using AQE reranking.
    Args:
        query_embedding (np.ndarray): Query image embedding.
        db_embeddings (np.ndarray): Database image embeddings.
    Returns:
        list: Ranked indices.
    """
    if aqe_reranking is None:
        st.error("AQE reranking module not available.")
        return []
    
    try:
        scores = aqe_reranking(query_embedding, db_embeddings)
        return np.argsort(-scores).tolist()
    except Exception as e:
        st.error(f"Error in AQE reranking: {e}")
        return []

def compare_images_cas(query_embedding, db_embeddings):
    """
    Compare query image to database images using CAS reranking.
    Args:
        query_embedding (np.ndarray): Query image embedding.
        db_embeddings (np.ndarray): Database image embeddings.
    Returns:
        list: Ranked indices.
    """
    if cas_reranking is None:
        st.error("CAS reranking module not available.")
        return []
    
    try:
        dist = cas_reranking(query_embedding, db_embeddings, metric='euclidean', k1=6, k2=60, k3=70, k4=7, k5=80, device=DEVICE)
        return np.argsort(dist).tolist()
    except Exception as e:
        st.error(f"Error in CAS reranking: {e}")
        return []

def compare_images_cs_hp(query_embedding, db_embeddings):
    """
    Placeholder for CS + HP ranking.
    Args:
        query_embedding (np.ndarray): Query image embedding.
        db_embeddings (np.ndarray): Database image embeddings.
    Returns:
        list: Ranked indices (using cosine similarity as fallback).
    """
    st.warning("CS + HP not implemented. Using cosine similarity as fallback.")
    return compare_images_cosine(query_embedding, db_embeddings)