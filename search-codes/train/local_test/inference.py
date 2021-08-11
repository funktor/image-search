import os
import base64
import requests

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list

data_dir = '/Users/amondal/Documents/datasets/Natural Images/natural_images'

filenames = sorted(get_file_list(data_dir))

image_path = random.choice(filenames)
b64_image = ""

with open(image_path, "rb") as imageFile:
    b64_image = base64.b64encode(imageFile.read())

data = {'b64': b64_image}
r = requests.post(url="http://localhost:8080/invocations", data=data)

h = r.json()
out = h['response']

imgdata = base64.b64decode(bytes(out, 'utf-8'))
filename = 'prediction.jpg'

with open(filename, 'wb') as f:
    f.write(imgdata)