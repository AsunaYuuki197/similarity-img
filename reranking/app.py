import os
import time
import numpy as np
from pandas import read_pickle
from PIL import Image
import streamlit as st
import torch

from revisitop import configdataset
from reranker.qe_reranking import aqe_reranking
from reranker.cas_reranking import cas_reranking

# Constants
MODEL_NAME = 'gl18-tl-resnet101-gem-w'
DATASET = 'roxford5k'
DATA_ROOT = '../dataset'
FEATURE_NAME = MODEL_NAME
TOP_K = 25
NUM_COLS = 5  # images per row
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# App title
st.title("Image Retrieval and Comparison")

# Load dataset config and features
cfg = configdataset(DATASET, DATA_ROOT)
features = read_pickle(os.path.join(DATA_ROOT, DATASET, f'features/{FEATURE_NAME}.pkl'))
gnd = read_pickle(os.path.join('..', 'groundtruth', f'gnd_{DATASET}.pkl'))

# Upload image
uploaded_image = st.file_uploader("Upload an image to compare:", type=["png", "jpg", "jpeg"])
embedding = None

def display_top_images(title, ranks, query_index, message=None):
    st.subheader(title)

    # Get ground truth relevant indices for the query
    gt_entry = gnd['gnd'][query_index]
    relevant_set = set(gt_entry['easy'] + gt_entry['hard'])

    # Count how many of the top-K retrieved are relevant
    top_k_ranks = ranks[:TOP_K]
    correct_count = sum(1 for idx in top_k_ranks if idx in relevant_set)

    if message:
        message += f" — {correct_count}/{TOP_K} correct"
        st.caption(message)

    for row_start in range(0, TOP_K, NUM_COLS):
        cols = st.columns(NUM_COLS)
        for col_idx in range(NUM_COLS):
            img_idx = row_start + col_idx
            if img_idx >= TOP_K:
                break

            rank_idx = ranks[img_idx]
            db_img_name = gnd['imlist'][rank_idx]
            db_img_path = os.path.join(DATA_ROOT, DATASET, 'jpg', f"{db_img_name}.jpg")

            try:
                img = Image.open(db_img_path).convert("RGB")

                # Decide border color
                is_relevant = rank_idx in relevant_set
                border_color = "green" if is_relevant else "red"
                caption = f"Rank {img_idx + 1}"

                # Create image with thick border
                border_thickness = 20
                bordered_img = Image.new(
                    "RGB",
                    (img.width + 2 * border_thickness, img.height + 2 * border_thickness),
                    border_color
                )
                bordered_img.paste(img, (border_thickness, border_thickness))

                cols[col_idx].image(bordered_img, caption=caption, use_container_width=True)

            except FileNotFoundError:
                cols[col_idx].error(f"Not found: {db_img_name}")



def show_ground_truth(title, indices, gnd, data_root, dataset):
    st.markdown(f"### {title}")
    cols = st.columns(min(len(indices), 5))
    for i, idx in enumerate(indices[:TOP_K]):  # Show up to 10 images
        img_name = gnd['imlist'][idx]
        img_path = os.path.join(data_root, dataset, 'jpg', f"{img_name}.jpg")
        img = Image.open(img_path).convert("RGB")
        cols[i % len(cols)].image(img, caption=f"{title} #{i+1}", use_container_width=True)



if uploaded_image:
    # Display uploaded image on the left with same width as result images
    st.subheader("Query Image")
    cols = st.columns(NUM_COLS)
    with cols[0]:
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)
    filename = uploaded_image.name.rsplit('.', 1)[0]

    # Try to get embedding from query or database
    try:
        query_index = gnd['qimlist'].index(filename)
        embedding = features['query'][query_index]
    except ValueError:
        try:
            query_index = gnd['imlist'].index(filename)
            embedding = features['db'][query_index]
        except ValueError:
            st.error(f"Filename '{filename}' not found in ground truth.")
            st.stop()

    # ground_truth = gnd['gnd'][index]

    # if 'easy' in ground_truth and ground_truth['easy']:
    #     show_ground_truth("Easy Ground Truth", ground_truth['easy'], gnd, DATA_ROOT, DATASET)

    # if 'medium' in ground_truth and ground_truth['medium']:
    #     show_ground_truth("Medium Ground Truth", ground_truth['medium'], gnd, DATA_ROOT, DATASET)

    # if 'hard' in ground_truth and ground_truth['hard']:
    #     show_ground_truth("Hard Ground Truth", ground_truth['hard'], gnd, DATA_ROOT, DATASET)



    # Cosine similarity
    start = time.time()
    cosine_scores = np.dot(embedding, features['db'].T)
    cosine_ranks = np.argsort(-cosine_scores)
    elapsed = time.time() - start
    display_top_images(
        f"Top {TOP_K} Similar Images (Cosine Similarity)",
        cosine_ranks,
        query_index,
        f"Executed in {elapsed:.4f} seconds"
    )

    # AQE reranking
    start = time.time()
    aqe_scores = aqe_reranking(embedding, features['db'])
    aqe_ranks = np.argsort(-aqe_scores)
    elapsed = time.time() - start
    display_top_images(
        f"Top {TOP_K} Similar Images (AQE Reranking)",
        aqe_ranks,
        query_index,
        f"Executed in {elapsed:.4f} seconds"
    )


    # CAS reranking
    start = time.time()
    dist = cas_reranking(embedding, features['db'], metric='euclidean', k1=6, k2=60, k3=70, k4=7, k5=80, device=DEVICE) # Using Cluster-Aware Similarity Algo
    cas_ranks = np.argsort(dist)
    elapsed = time.time() - start
    display_top_images(
        f"Top {TOP_K} Similar Images (CAS Reranking)",
        cas_ranks,
        query_index,
        f"Executed in {elapsed:.4f} seconds"
    )

