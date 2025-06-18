import numpy as np
import json
import os
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Convert text to CLIP embedding
def get_text_embedding(text):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return  text_features.squeeze().cpu().numpy()

# Convert image to CLIP embedding
def get_image_embedding(image):
    if isinstance(image, str):  # if it's a file path
        image = Image.open(image)

    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.squeeze().cpu().numpy()
