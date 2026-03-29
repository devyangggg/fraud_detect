import requests
import numpy as np

url = "http://localhost:8000/predict"
dummy = np.random.randn(100, 6, 3).tolist()
response = requests.post(url, json={"data": dummy})
print(response.json())