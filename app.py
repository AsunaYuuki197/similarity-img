import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'reranking')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'hp_and_cs')))


import numpy as np
from pandas import read_pickle
from PIL import Image
import streamlit as st
import torch

from reranking.revisitop import configdataset
from reranking.reranker.qe_reranking import aqe_reranking
from reranking.reranker.cas_reranking import cas_reranking
from hp_and_cs.hypergraph_propagation import prepare_hypergraph_propagation, propagate
from hp_and_cs.utils.retrieval_component import connect_nodup


# Constants
MODEL_NAME = 'delg_r50'
DATASET = 'roxford5k'
DATA_ROOT = 'dataset'
FEATURE_NAME = MODEL_NAME
TOP_K = 25
NUM_COLS = 5  # images per row
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HYPERGRAPH_PROPAGATION=1
COMMUNITY_SELECTION=2
HP_ROOT = 'hp_and_cs'

# App title
st.title("Image Retrieval and Comparison")

# Load dataset config and features
cfg = configdataset(DATASET, DATA_ROOT)
features = read_pickle(os.path.join(DATA_ROOT, DATASET, f'features/{FEATURE_NAME}.pkl'))
gnd = read_pickle(os.path.join('groundtruth', f'gnd_{DATASET}.pkl'))


#load the features for Hypergraph Propagation and Community Selection
if DATASET.startswith('roxford'):
    hp_vecs=np.load(f'{HP_ROOT}/features/roxford_np_delg_features/a_global_vecs.npy').T # (2048,4993)
    hp_qvecs=np.load(f'{HP_ROOT}/features/roxford_np_delg_features/a_global_qvecs.npy').T #(2048,70)

elif DATASET.startswith('rparis'):
    hp_vecs=np.load(f'{HP_ROOT}/features/rparis_np_delg_features/a_global_vecs.npy').T 
    hp_qvecs=np.load(f'{HP_ROOT}/features/rparis_np_delg_features/a_global_qvecs.npy').T 
    
elif DATASET.startswith('R1Moxford'):
    hp_distractors=np.load(f'{HP_ROOT}/features/distractor_np_delg_features/a_global_vecs.npy').T
    hp_vecs=np.load(f'{HP_ROOT}/features/roxford_np_delg_features/a_global_vecs.npy').T # (2048,4993)
    hp_vecs=np.concatenate((hp_vecs,hp_distractors),axis=1) #()
    del hp_distractors
    hp_qvecs=np.load(f'{HP_ROOT}/features/roxford_np_delg_features/a_global_qvecs.npy').T #(2048,70) 
    
elif DATASET.startswith('R1Mparis'):
    hp_distractors=np.load(f'{HP_ROOT}/features/distractor_np_delg_features/a_global_vecs.npy').T
    hp_vecs=np.load(f'{HP_ROOT}/features/rparis_np_delg_features/a_global_vecs.npy').T # (2048,6322)
    hp_vecs=np.concatenate((hp_vecs,hp_distractors),axis=1) #()
    hp_qvecs=np.load(f'{HP_ROOT}/features/rparis_np_delg_features/a_global_qvecs.npy').T #(2048,70)  

#prepare the hypergraph propagation
if HYPERGRAPH_PROPAGATION==True:
    prepare_hypergraph_propagation(HP_ROOT, DATASET)

#prepare the community selection
if COMMUNITY_SELECTION>=True:
    from hp_and_cs.community_selection import prepare_community_selection,extract_sub_graph,calculate_entropy,match_one_pair_delg,find_dominant
    
    prepare_community_selection(HP_ROOT, DATASET, COMMUNITY_SELECTION)

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



def rank_with_hp_cs(hp_qranks, query_index,
                    COMMUNITY_SELECTION=1,
                    HYPERGRAPH_PROPAGATION=True):
    """
    Runs first retrieval, optional community selection, and optional hypergraph propagation.

    Args:
        hp_qranks (np.ndarray): Ranking matrix where each column corresponds to a query's ranked images.
        query_index (int): Index of the current query.
        q (str): Query image name or identifier.
        COMMUNITY_SELECTION (int or bool): 
            0 or False = skip, 
            1 = only calculate uncertainty, 
            2 = also apply dominant selection based on inlier threshold.
        HYPERGRAPH_PROPAGATION (bool): Whether to apply hypergraph propagation.

    Returns:
        final_ranks (list): Ranked list of image identifiers.
        dominant_image (str): Dominant image used as the root.
    """
    # First search
    if hp_qranks.ndim == 1:
        first_search = list(hp_qranks)  # 1D: already ranks for a single query
    else:
        first_search = list(hp_qranks[:, query_index])  # 2D: multiple queries

    dominant_image = first_search[0]

    # Community selection
    if COMMUNITY_SELECTION:
        Gs = extract_sub_graph(first_search, 20)
        uncertainty = calculate_entropy(Gs)

        if COMMUNITY_SELECTION == 2 and uncertainty > 1:
            inlier, size = match_one_pair_delg(query_index, first_search[0])
            if inlier < 20:
                Gs = extract_sub_graph(first_search, 100)
                dominant_image = find_dominant(Gs, first_search, query_index)
        else:
            print(f"Uncertainty: {uncertainty}")

    # Hypergraph propagation
    if HYPERGRAPH_PROPAGATION:
        rank_list = propagate([dominant_image])
        final_ranks = connect_nodup(rank_list, first_search)
    else:
        final_ranks = first_search

    return final_ranks, dominant_image






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


    # CS + SV + HG
    hp_qscores=np.dot(hp_vecs.T, hp_qvecs[:, query_index]) 
    hp_qranks=np.argsort(-hp_qscores)
    start = time.time()

    hp_final_ranks, hp_dominant = rank_with_hp_cs(
        hp_qranks, query_index=query_index, 
        COMMUNITY_SELECTION=COMMUNITY_SELECTION, 
        HYPERGRAPH_PROPAGATION=HYPERGRAPH_PROPAGATION
    )
    elapsed = time.time() - start
    display_top_images(
        f"Top {TOP_K} Similar Images (CS + SV + HP)",
        hp_final_ranks,
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



    
    

