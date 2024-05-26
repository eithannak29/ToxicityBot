import requests
import os
from dotenv import load_dotenv
import random

load_dotenv()

TOKEN_HUG=os.getenv('TOKEN_HUG')

API_URL = "https://api-inference.huggingface.co/models/zcamz/bert-finetuned-toxic"
headers = {"Authorization": f"Bearer {TOKEN_HUG}"}

def detect_language(text):
    url = "https://libretranslate.de/detect"
    payload = {'q': text}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=payload, headers=headers)
    result = response.json()
    return result[0]['language']

def translate_text(text, target_language='en'):
    source_language = detect_language(text)
    if source_language == target_language:
        return text
    url = "https://libretranslate.de/translate"
    payload = {
        'q': text,
        'source': source_language,
        'target': target_language,
        'format': 'text'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=payload, headers=headers)
    result = response.json()
    return result['translatedText']


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response_json = response.json()
    # return response_json
    return {item['label']: item['score'] for item in response_json[0]}


def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# output = query({
#     "inputs": "I hate toy.",
# })

# print(output)