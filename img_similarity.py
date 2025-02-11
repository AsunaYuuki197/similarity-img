import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from io import BytesIO
from bs4 import BeautifulSoup

st.title("Image Comparison App")

# get img links
## from disk
def get_image_links_from_disk(directory):
    image_paths = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('png', 'jpg', 'jpeg'))]
    return image_paths

## from cloud storage (Google Drive)
def get_image_links_from_drive(folder_link):
    url = f"{folder_link}"

    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to access the folder.")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    image_paths = []
    image_link = {}
    for script in soup.find_all("script"):
        if "https://drive.google.com/file/d/" in script.text:
            lines = script.text.split(',')
            for line in lines:
                if "https://drive.google.com/file/d/" in line and line not in image_link.keys():
                   
                    image_link[line] = 1
                    response = requests.get(line[1:-1])
                    if response.status_code != 200:
                        print("Failed to access the image.")
                        break

                    tmp_soup = BeautifulSoup(response.text, 'html.parser')

                    for tmp_script in tmp_soup.find_all("script"):
                        tmp_lines = tmp_script.text.split(',')
                        for tmp_line in tmp_lines:
                            if "drive.google.com/drive-viewer" in tmp_line and tmp_line not in image_paths:
                                image_paths.append(tmp_line[1:-1].replace("\\u003d", "="))
                                break
    
    
    return image_paths


# load image from disk or cloud
def load_image(image_link):
    if image_link.startswith(("http", "https")):
        response = requests.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_link)
    return image


# compare by image matrix 
def compare_images_matrix(img1, img2):
    img1 = np.array(img1.resize((224, 224))) 
    img2 = np.array(img2.resize((224, 224)))

    if len(img1.shape) > 2 and img1.shape[2] == 4:  # RGBA to RGB
        img1 = img1[:, :, :3]
    if len(img2.shape) > 2 and img2.shape[2] == 4:  # RGBA to RGB
        img2 = img2[:, :, :3]

    if len(img1.shape) == 2:  # img1 is grayscale
        img1 = np.stack([img1] * 3, axis=-1)
    if len(img2.shape) == 2:  # img1 is grayscale
        img2 = np.stack([img2] * 3, axis=-1)

    img1 = img1 / 255.0
    img2 = img2 / 255.0

    difference = np.mean(np.abs(img1 - img2))
    return difference


# compare by image context - feature (still in developing)


# Image source selection
source = st.radio("Select image source:", ("From Disk", "From Cloud Storage"))
if 'image_links' not in st.session_state:
    st.session_state.image_links = []

# Get image links based on the source
if source == "From Disk":
    directory = st.text_input("Enter directory path to load images from disk (e.g: C:\\Users\\ACER\\Pictures):")
    if st.button("Load"):
        st.session_state.image_links = get_image_links_from_disk(directory)
else:
    directory = st.text_input("Enter folder link to load images from google drive (e.g: https://drive.google.com/drive/folders/{folder_id}):")
    if st.button("Load"):
        st.session_state.image_links = get_image_links_from_drive(directory)


# Display image links
if st.session_state.image_links:
    st.write("Images loaded success")
else:
    st.write("Images loaded fail") 
# Upload an image to compare
uploaded_image = st.file_uploader("Upload an image to compare:", type=["png", "jpg", "jpeg"])


# Image comparison
if st.button("Compare") and   st.session_state.image_links and uploaded_image is not None:
    img1 = Image.open(uploaded_image)
    st.image(img1, caption="Uploaded Image", use_container_width=True)

    st.write("Comparing to images from the selected source...")

    results = []
    for link in st.session_state.image_links:
        img2 = load_image(link)

        # Compare by matrix
        matrix_diff = compare_images_matrix(img1, img2)
        
        if matrix_diff > 0.1:
            continue

        # Compare by embedding (still in developing)

        results.append((link, matrix_diff))

    st.write("Comparison complete. Displaying from the selected source...")    
    # Return result image + image links
    best_match = sorted(results, key=lambda x: x[1], reverse=True)
    for i in best_match:
        result_img = load_image(i[0])
        st.image(result_img, caption=f"{i[0]}", use_container_width=True)
    
    st.write("Finished")


