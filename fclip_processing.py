from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import torch
import torch.nn.utils.rnn as r
from .image_processing import get_img

model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
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
        datas = processor(images=[img], return_tensors='pt').to(device)

        with torch.no_grad():
            embeddings = model.get_image_features(**datas).to(device)

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
    datas = processor(text, padding="max_length", return_tensors="pt", 
                       max_length=77, truncation=True).to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(**datas).to(device)

    return embeddings


