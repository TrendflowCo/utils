from io import BytesIO
from PIL import Image
import requests
import time
import sys
sys.path.append('..')
from .scraping import generate_user_agent

def get_img(url_image, cookies_dict=None, timeout=60):
    """
    Get the image from the URL.
    
    Args:
        url_image (str): The URL of the image
        cookies_dict (dict): The cookies to use for the request
        timeout (int): The timeout in seconds
    
    Returns:
        img (PIL.Image.Image): The image
    """
    if url_image is not None:
        try:
            # Randomly select a User-Agent for each request
            user_agent = generate_user_agent()
            
            # Set headers including User-Agent
            headers = {
                'User-Agent': user_agent
            }
            
            # Create a session to manage cookies
            with requests.Session() as session:
                if cookies_dict:
                    response = session.get(url_image, 
                                           cookies=cookies_dict,
                                           headers=headers, 
                                           timeout=timeout)
                else:
                    response = session.get(url_image, 
                                           headers=headers, 
                                           timeout=timeout)
                response.raise_for_status()  # Raise an exception for bad responses
                
                # Add a delay between requests (optional)
                time.sleep(1)
                
                img = Image.open(BytesIO(response.content)).convert("RGB")
                return img
        except requests.exceptions.RequestException as e:
            pass
            # print(f"Error: {e}")
    
    return None  # Return None if there was an issue
