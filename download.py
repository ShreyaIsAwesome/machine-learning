import gdown

url = 'https://drive.google.com/uc?id=1rPrwRTzHqDzMj_2-EcMLgHWvg61tHg_-'
output = 'model/model.keras'
gdown.download(url, output, quiet=False)