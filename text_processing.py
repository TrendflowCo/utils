import re
import pandas as pd

def generate_prompts(row):
    prompts = []
    tag = row['value']
    category = row['category']
    for tag_ in [tag, f"{tag} {category.replace('_', ' ')}"]:
        for template in templates_tags:
            tag_template = template.format(tag_)
            prompts.append(tag_template)
    return prompts


def build_text(item_data, cols):
    if item_data is not None:
        text = ''
        for col in cols:
            if pd.notnull(item_data[col]):
                text += clean_text(item_data[col])
                text += ' '
        return text.strip().lower()
        
def remove_extra_spaces(text):
    # Split the text by spaces and filter out empty strings, then join back with a single space
    return ' '.join(word for word in text.split(' ') if word)

def clean_desc(raw_text):
    text = '. '.join([x.strip() for x in raw_text.split('.')]).strip().lower()
    text = text.replace('we work with monitoring programs to guarantee compliance with the social, environmental, and health and safety standards of our products. to assess its compliance, we have developed an audit program and plans for continual improvement.', '')
    text = text.replace('<br>', '').replace('#','').replace('%', '')
    text = ''.join([i for i in text if not i.isdigit()])
    return remove_extra_spaces(text)
    
def clean_text(text):
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
    full_text = ''
    for col in ['name_en', 'desc_1_en', 'desc_2_en']:
        full_text += ' ' 
        full_text += clean_text(row[col])
    
    full_text += f" Made by {row['brand']}."
    full_text += f" For {row['category']}."
    
    # currency_text = ' euros' if row['currency']=='EUR' else ' dollars' if row['currency']=='USD' else ''
    # if pd.notnull(row['price']):
    #     full_text += f" It's price is {str(int(row['price']))}"
    #     full_text += currency_text+'.'
    # # full_text += f" This item is{'' if row['sale'] else ' not'} on sale"
    #     full_text += f" This item is on sale" if row['sale'] else ''
    full_text = full_text.strip()
    
    return full_text


def create_full_text_2(row):
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
        