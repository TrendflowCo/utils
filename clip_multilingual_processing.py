import torch
import requests
from sentence_transformers import SentenceTransformer
from .image_processing import get_img

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_model.to(device)

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

def compute_image_embeddings(url_image, cookies_dict=None):
    """
    Compute the image embeddings for a given image.
    
    Args:
        url_image (str): The URL of the image
        cookies_dict (dict): The cookies to use for the request
    
    Returns:
        image_features (torch.Tensor): The image features of shape (1, 512)
    """
    try:
        img = get_img(url_image, cookies_dict)

        with torch.no_grad():
            embeddings = img_model.encode([img],  convert_to_tensor=True,)

        return embeddings
    
    except Exception as e:
        print(e)
        return None

        
def compute_text_embeddings(text):
    """
    Compute the text embeddings for a given text.
    
    Args:
        text (str): The text to embed
    
    Returns:
        text_features (torch.Tensor): The text features of shape (1, 512)
    """
    if isinstance(text, str):
        text = [text]
    with torch.no_grad():
        embeddings = text_model.encode(text,  convert_to_tensor=True).to(device)

    return embeddings


