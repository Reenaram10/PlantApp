import requests
import json
import os
import pprint

def test_identify():
    url = "http://127.0.0.1:8082/api/identify"
    img_path = os.path.join("plant_disease_detection", "image.png")
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return
        
    print(f"Sending POST request to {url} with image {img_path}...")
    with open(img_path, 'rb') as f:
        files = {'image': f}
        r = requests.post(url, files=files)
        
    print(f"Status Code: {r.status_code}")
    try:
        res = r.json()
        print("\nAPI Response:")
        pprint.pprint(res)
    except Exception as e:
        print("Failed to decude JSON:", e)
        print("Raw response:", r.text)

if __name__ == "__main__":
    test_identify()
