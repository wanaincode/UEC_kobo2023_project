from openai import OpenAI
from dotenv import load_dotenv
import base64
import requests
import os 

load_dotenv()

# Set model name and API key
MODEL = 'gpt-4o'
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

IMAGE_PATH = 'test_item.jpg'
base64_image = encode_image(IMAGE_PATH)


response = client.chat.completions.create(
  model= MODEL,
  messages=[
    {
      "role": "user",
      "content": [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"}
        },{
            "type": "text",
            "text": "Send back name and calorie of this item using certain form below:\nName:\nCalorie:"
        }
    ]},
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
# Name and Calorie of the food item by API
obj_info_byOPENAI = response.choices[0].message.content.split('\n')
print(obj_info_byOPENAI)
obj_name = obj_info_byOPENAI[0]
obj_calorie = obj_info_byOPENAI[1]
print(f'{obj_name}, {obj_calorie}')