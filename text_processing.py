import re
import pandas as pd

def generate_prompts(row):
    """
    Generate the prompts for a given row.
    
    Args:
        row (pd.Series): The row of the dataframe
    
    Returns:
        prompts (list): The prompts for the row
    """
    prompts = []
    tag = row['value']
    category = row['category']
    for tag_ in [tag, f"{tag} {category.replace('_', ' ')}"]:
        for template in templates_tags:
            tag_template = template.format(tag_)
            prompts.append(tag_template)
    return prompts


def build_text(item_data, cols):
    """
    Build the text to use for the text embeddings.
    
    Args:
        item_data (dict): The item data
        cols (list): The columns to use for the text embeddings
    
    Returns:
        text (str): The text to use for the text embeddings
    """
    if item_data is not None:
        text = ''
        for col in cols:
            if pd.notnull(item_data[col]):
                text += clean_text(item_data[col])
                text += ' '
        return text.strip().lower()
        
def remove_extra_spaces(text):
    """
    Remove extra spaces from a given text.
    
    Args:
        text (str): The text to remove extra spaces from
    
    Returns:
        text (str): The text with extra spaces removed
    """
    if text == '' or text is None:
        return ''
    # Split the text by spaces and filter out empty strings, then join back with a single space
    return ' '.join(word for word in text.split(' ') if word)

def clean_desc(raw_text):
    """
    Clean the description text.
    
    Args:
        raw_text (str): The raw description text
    
    Returns:
        text (str): The cleaned description text
    """
    if raw_text == '' or raw_text is None:
        return ''
    text = '. '.join([x.strip() for x in raw_text.split('.')]).strip().lower()
    text = text.replace('we work with monitoring programs to guarantee compliance with the social, environmental, and health and safety standards of our products. to assess its compliance, we have developed an audit program and plans for continual improvement.', '')
    text = text.replace('<br>', '').replace('#','').replace('%', '')
    text = ''.join([i for i in text if not i.isdigit()])
    text = remove_extra_spaces(text)
    return text
    
def clean_text(text):
    """
    Clean a given text.
    
    Args:
        text (str): The text to clean
    
    Returns:
        clean_text (str): The cleaned text
    """
    if text == '' or text is None:
        return ''
    clean_text = text
    clean_text = clean_text.strip()
    clean_text = clean_text.lower()
    clean_text = clean_text.replace('\n', '')
    
    clean_text = clean_text.replace('t-shirt', '_TSHIRT_')
    clean_text = clean_text.replace('-', '. ')
    clean_text = clean_text.replace('_TSHIRT_', 't-shirt')
    
    if clean_text.startswith('.'):
        clean_text = clean_text[1:]
    if len(clean_text)>0:
        if clean_text[-1] != '.':
            clean_text = clean_text + '.'
    clean_text = re.sub(' +', ' ', clean_text)

    clean_text = clean_text.strip()
    clean_text = clean_text.title()
    return clean_text


def create_full_text(row):
    """
    Create the full text for a given row.
    
    Args:
        row (pd.Series): The row of the dataframe
    
    Returns:
        full_text (str): The full text for the row
    """
    full_text = ''
    for col in ['name_en', 'desc_1_en', 'desc_2_en']:
        full_text += ' ' 
        full_text += clean_text(row[col])
    
    full_text += f" Made by {row['brand']}."
    full_text += f" For {row['category']}."
    full_text = full_text.strip()
    
    return full_text


def create_full_text_2(row):
    """
    Create the full text for a given row.
    
    Args:
        row (pd.Series): The row of the dataframe
    
    Returns:
        full_text (str): The full text for the row
    """
    full_text = ''
    for col in ['name_en']:
        full_text += ' ' 
        full_text += clean_text(row[col])
    
    full_text += f" Made by {row['brand']}."
    full_text += f" For {row['category']}."
    
    if pd.notnull(row['price']):
        full_text += f" On sale." if row['sale'] else ''
    full_text = full_text.strip()
    return full_text

def get_text_embeddings_without_dataset(text):
    """
    Get the text embeddings for a given text.
    
    Args:
        text (str): The text to embed
    
    Returns:
        embeddings (torch.Tensor): The text embeddings
    """
    try:
        if isinstance(text, str):
            text = [text]
        datas = processor(text, padding="max_length", return_tensors="pt", 
                           max_length=77, truncation=True)
        with torch.no_grad():
            embeddings = model.get_text_features(**datas)

        return embeddings
    except Exception as e:
        print(e)
        