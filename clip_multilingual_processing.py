import torch
import requests
from sentence_transformers import SentenceTransformer
from .image_processing import get_img

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

def compute_image_embeddings(url_image, cookies_dict=None):
    try:
        img = get_img(url_image, cookies_dict)

        with torch.no_grad():
            embeddings = img_model.encode([img],  convert_to_tensor=True,)

        return embeddings
    
    except Exception as e:
        print(e)
        return None

        
def compute_text_embeddings(text):
    if isinstance(text, str):
        text = [text]
    with torch.no_grad():
        embeddings = text_model.encode(text,  convert_to_tensor=True)

    return embeddings


