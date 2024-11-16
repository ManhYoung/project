import requests

url = "http://127.0.0.1:5000/predict"
file = {"file": open("6598d808aad49302723348__1200.jpg", "rb")}

response = requests.post(url, files=file)
print(response.json())
