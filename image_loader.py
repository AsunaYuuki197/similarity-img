import os
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image
import streamlit as st
from config import VALID_EXTENSIONS, REQUEST_TIMEOUT

def get_image_links_from_disk(directory):
    """
    Retrieve all image file paths from a directory, including subdirectories.
    Args:
        directory (str): Path to the directory containing images.
    Returns:
        list: List of valid image file paths.
    """
    if not directory or not isinstance(directory, str):
        st.error("Directory path must be a non-empty string.")
        return []
    
    if not os.path.isdir(directory):
        st.error(f"Invalid directory: {directory}")
        return []
    
    image_paths = []
    try:
        for root, _, files in os.walk(directory):
            for file in sorted(files):  # Sort for consistent ordering
                if file.lower().endswith(VALID_EXTENSIONS):
                    image_paths.append(os.path.join(root, file))
        if not image_paths:
            st.warning(f"No images found in directory: {directory}")
    except PermissionError:
        st.error(f"Permission denied accessing directory: {directory}")
    except Exception as e:
        st.error(f"Error accessing directory {directory}: {e}")
    
    return image_paths

def get_image_links_from_drive(folder_link: str) -> list[str]:
    """
    Retrieve image URLs from a Google Drive folder link.

    Args:
        folder_link (str): Google Drive folder URL.
    Returns:
        list[str]: List of image URLs.
    """
    if not folder_link or not isinstance(folder_link, str):
        st.error("Google Drive folder link must be a non-empty string.")
        return []

    if not folder_link.startswith("https://drive.google.com"):
        st.error("Invalid Google Drive folder link.")
        return []

    st.info("Attempting to fetch images from Google Drive folder...")

    try:
        response = requests.get(folder_link, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            st.error(f"Failed to access Google Drive folder: HTTP {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        image_urls, seen = [], set()

        # Extract file links
        for script in soup.find_all("script"):
            script_text = script.get_text()
            if "https://drive.google.com/file/d/" not in script_text:
                continue

            for line in script_text.split(","):
                if "https://drive.google.com/file/d/" in line:
                    link = line.strip().strip('"').strip("'")
                    if link in seen:
                        continue
                    seen.add(link)

                    # Open file page
                    file_res = requests.get(link, timeout=REQUEST_TIMEOUT)
                    if file_res.status_code != 200:
                        continue

                    tmp_soup = BeautifulSoup(file_res.text, "html.parser")

                    # Extract viewer links (direct image access)
                    for tmp_script in tmp_soup.find_all("script"):
                        for tmp_line in tmp_script.get_text().split(","):
                            if "drive.google.com/drive-viewer" in tmp_line:
                                clean_url = tmp_line.strip().strip('"').strip("'").replace("\\u003d", "=")
                                image_urls.append(clean_url)
                                break

        if not image_urls:
            st.warning("No valid image URLs found in the Google Drive folder.")
        return image_urls

    except requests.Timeout:
        st.error("Request to Google Drive timed out.")
        return []
    except Exception as e:
        st.error(f"Unexpected error while parsing folder: {e}")
        return []

def load_image(image_link):
    """
    Load an image from a file path or URL.
    Args:
        image_link (str): Path or URL to the image.
    Returns:
        PIL.Image: Loaded image or None if failed.
    """
    try:
        if isinstance(image_link, str) and image_link.startswith(("http", "https")):
            response = requests.get(image_link, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                st.error(f"Failed to load image from {image_link}: HTTP {response.status_code}")
                return None
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_link).convert("RGB")
        return image
    except requests.Timeout:
        st.error(f"Request to {image_link} timed out.")
        return None
    except Exception as e:
        st.error(f"Error loading image {image_link}: {e}")
        return None