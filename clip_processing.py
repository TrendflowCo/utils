import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as r
import clip
import numpy as np
from .image_processing import get_img

model, preprocess = clip.load("ViT-L/14@336px")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)


def compute_image_embeddings(url_image, cookies_dict=None):
    try:
        img = get_img(url_image, cookies_dict)
        img = preprocess(img)
        img_tensor = r.pad_sequence([img],  batch_first=True).to(device) #maybe np.stack?
        with torch.no_grad():
            image_features = model.encode_image(img_tensor).float()
        return image_features
    except Exception as e:
        print(e)
        return None

def compute_text_embeddings(text):
    try:
        text = clip.tokenize(text, context_length=77, truncate=True).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text).to(device)
        return text_features
    except Exception as e:
        print(e)
        return None
