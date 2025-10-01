import os

import streamlit as st
from config import TOP_K, NUM_COLS
from image_loader import load_image

def display_top_images(title, ranks, image_links, message=None):
    """
    Display ranked images in a grid.
    Args:
        title (str): Title for the section.
        ranks (list): Indices of ranked images.
        image_links (list): List of image paths/URLs.
        message (str): Optional message to display.
    """
    st.subheader(title)
    if message:
        st.caption(message)

    for row_start in range(0, min(len(ranks), TOP_K), NUM_COLS):
        cols = st.columns(NUM_COLS)
        for col_idx in range(NUM_COLS):
            img_idx = row_start + col_idx
            if img_idx >= len(ranks) or img_idx >= TOP_K:
                break

            rank_idx = ranks[img_idx]
            if rank_idx >= len(image_links):
                continue
            img_path = image_links[rank_idx]
            img = load_image(img_path)
            if img is None:
                continue

            caption = f"Rank {img_idx + 1}: {os.path.basename(img_path)}"
            cols[col_idx].image(img, caption=caption, use_container_width=True)