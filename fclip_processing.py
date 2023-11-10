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
    try:
        img = get_img(url_image, cookies_dict)
        datas = processor(images=[img], return_tensors='pt').to(device)

        with torch.no_grad():
            embeddings = model.get_image_features(**datas).to(device)

        return embeddings
    
    except Exception as e:
        print(e)
        return None
    
# def get_image_embeddings_with_dataset(url_image):

#     try:
#         img = get_img(url_image)
#         img = pass_images_through_data([img])
#         datas = processor(images=img, return_tensors='pt')
        
#         with torch.no_grad():
#             embeddings = model.get_image_features(**datas)

#         return embeddings.numpy()
#     except Exception as e:
#         print(e)
#         return None


        
def compute_text_embeddings(text):
    if isinstance(text, str):
        text = [text]
    datas = processor(text, padding="max_length", return_tensors="pt", 
                       max_length=77, truncation=True).to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(**datas).to(device)

    return embeddings


