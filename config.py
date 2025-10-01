import torch

TOP_K = 25  # Number of top images to display
NUM_COLS = 5  # Number of images per row in the UI
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MATRIX_THRESHOLD = 0.1
FEATURE_THRESHOLD = 0.85
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
REQUEST_TIMEOUT = 10  # Timeout for HTTP requests in seconds