import streamlit as st
import numpy as np
import time
from PIL import Image
from image_loader import get_image_links_from_disk, get_image_links_from_drive, load_image
from image_comparator import (
    compare_images_matrix,
    compare_images_feature,
    compare_images_cosine,
    compare_images_aqe,
    compare_images_cas,
    compare_images_cs_hp
)
from utils import display_top_images
from config import TOP_K, MATRIX_THRESHOLD, FEATURE_THRESHOLD, NUM_COLS

def main():
    st.title("Image Comparison App")

    # Image source selection
    source = st.radio("Select image source:", ("From Disk", "From Google Drive"))
    if 'image_links' not in st.session_state:
        st.session_state.image_links = []

    # Get image links based on the source
    if source == "From Disk":
        directory = st.text_input("Enter directory path to load images (e.g., C:\\Users\\ACER\\Pictures):")
        if st.button("Load Images"):
            st.session_state.image_links = get_image_links_from_disk(directory)
    else:
        directory = st.text_input("Enter Google Drive folder link (e.g., https://drive.google.com/drive/folders/{folder_id}):")
        if st.button("Load Images"):
            st.session_state.image_links = get_image_links_from_drive(directory)

    # Display image links status
    if st.session_state.image_links:
        st.success(f"Loaded {len(st.session_state.image_links)} images successfully.")
    else:
        st.warning("No images loaded.")

    # Upload query image
    uploaded_image = st.file_uploader("Upload an image to compare:", type=["png", "jpg", "jpeg"])

    # Image comparison method selection
    compare_options = ["Matrix", "Feature", "Cosine Similarity"]
    compare_options.append("AQE Reranking")
    compare_options.append("CAS Reranking")
    compare_options.append("CS + HP")
    compare = st.radio("Compare by:", compare_options)

    if st.button("Compare") and uploaded_image and st.session_state.image_links:
        query_img = load_image(uploaded_image)
        if query_img is None:
            st.error("Failed to load query image.")
            return

        st.subheader("Query Image")
        cols = st.columns(NUM_COLS)
        with cols[0]:
            st.image(query_img, caption=uploaded_image.name, use_container_width=True)

        st.write("Comparing to images from the selected source...")

        # Load database image embeddings
        db_embeddings = []
        valid_image_links = []
        for link in st.session_state.image_links:
            img = load_image(link)
            if img is None:
                continue
            from get_embedding import get_image_embedding
            embedding = get_image_embedding(img)
            db_embeddings.append(embedding)
            valid_image_links.append(link)
        
        if not valid_image_links:
            st.error("No valid images loaded from the source.")
            return
        db_embeddings = np.array(db_embeddings)

        # Get query image embedding
        from get_embedding import get_image_embedding
        query_embedding = get_image_embedding(query_img)

        ranks = []
        start = time.time()
        if compare == "Matrix":
            results = []
            for idx, link in enumerate(valid_image_links):
                img = load_image(link)
                if img is None:
                    continue
                score = compare_images_matrix(query_img, img)
                if score <= MATRIX_THRESHOLD:
                    results.append((idx, score))
            results.sort(key=lambda x: x[1])  # Sort by difference (lower is better)
            ranks = [idx for idx, _ in results]
            elapsed = time.time() - start
            display_top_images("Top Images (Matrix Comparison)", ranks, valid_image_links, f"Executed in {elapsed:.4f} seconds")

        elif compare == "Feature":
            results = []
            for idx, link in enumerate(valid_image_links):
                img = load_image(link)
                if img is None:
                    continue
                score = compare_images_feature(query_img, img)
                if score >= FEATURE_THRESHOLD:
                    results.append((idx, score))
            results.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity (higher is better)
            ranks = [idx for idx, _ in results]
            elapsed = time.time() - start
            display_top_images("Top Images (Feature Comparison)", ranks, valid_image_links, f"Executed in {elapsed:.4f} seconds")

        elif compare == "Cosine Similarity":
            ranks = compare_images_cosine(query_embedding, db_embeddings)
            elapsed = time.time() - start
            display_top_images("Top Images (Cosine Similarity)", ranks, valid_image_links, f"Executed in {elapsed:.4f} seconds")

        elif compare == "AQE Reranking":
            ranks = compare_images_aqe(query_embedding, db_embeddings)
            elapsed = time.time() - start
            display_top_images("Top Images (AQE Reranking)", ranks, valid_image_links, f"Executed in {elapsed:.4f} seconds")

        elif compare == "CAS Reranking":
            ranks = compare_images_cas(query_embedding, db_embeddings)
            elapsed = time.time() - start
            display_top_images("Top Images (CAS Reranking)", ranks, valid_image_links, f"Executed in {elapsed:.4f} seconds")

        elif compare == "CS + HP":
            ranks = compare_images_cs_hp(query_embedding, db_embeddings)
            elapsed = time.time() - start
            display_top_images("Top Images (CS + HP)", ranks, valid_image_links, f"Executed in {elapsed:.4f} seconds")

        if ranks:
            st.success("Comparison complete.")
        else:
            st.warning("No similar images found.")
    else:
        if not uploaded_image:
            st.warning("Please upload an image to compare.")
        if not st.session_state.image_links:
            st.warning("Please load images from a source.")

if __name__ == "__main__":
    main()