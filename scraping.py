import requests
import random
import string


def prepare_datasets(df, necessary_cols):

    for col in ['img_url', 'shop_link']:
        df = df[~df[col].isna()]

    items = df[[col for col in necessary_cols if col!='img_url']]
    
    images = df[['shop_link', 'img_url']].drop_duplicates().reset_index(drop=True)

    # Count the number of non-null values in each row
    items['non_null_count'] = items.notnull().sum(axis=1)

    # Sort the dataframe by non-null count in descending order
    items = items.sort_values('non_null_count', ascending=False)

    # Drop duplicated rows keeping the first occurrence based on priority columns
    #priority_columns = ['shop_link', 'name', 'desc_1', 'price', 'category', 'old_price']
    items = items.drop_duplicates(subset='shop_link', keep='first')

    # Remove the 'non_null_count' column
    items = items.drop('non_null_count', axis=1).reset_index(drop=True)

    return items, images

    
def get_cookies(url):
    
    headers = {
            'User-Agent': generate_user_agent()
    }
    
    # Send an initial HTTP request to the website
    response = requests.get(url, headers=headers)

    # Retrieve the cookies from the response
    cookies = response.cookies

    # Convert the cookies to a dictionary
    cookies_dict = {cookie.name: cookie.value for cookie in cookies}
    
    return cookies_dict


def generate_user_agent():
    # Generate a random browser name
    browser = ''.join(random.choices(string.ascii_uppercase, k=5))
    
    # Generate a random browser version
    version_major = random.randint(1, 20)
    version_minor = random.randint(0, 9)
    version_build = random.randint(1000, 9999)
    browser_version = f"{version_major}.{version_minor}.{version_build}"
    
    # Generate a random platform
    platforms = ['Windows NT', 'Macintosh', 'X11', 'Linux']
    platform = random.choice(platforms)
    
    # Generate a random language
    languages = ['en-US', 'en-GB', 'es-ES', 'fr-FR', 'de-DE']
    language = random.choice(languages)
    
    # Generate a random string of letters and digits
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    
    # Generate a random user agent
    user_agent = f"Mozilla/5.0 ({platform}; {random_string}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{browser_version} Safari/537.36 {language}"
    
    return user_agent