from skimage.transform import resize
import numpy as np
# from keras.models import load_model

from PIL import Image
import requests
from io import BytesIO


def preprocess_image(img):
    size = (28, 28, 1)
    # image to grayscale
    img = np.array(img)
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    # image resize
    img = resize(img, size)
    return img


def predict_digit(img, model):
    test_img = preprocess_image(img)[None, ...]  # model input - ( bs, h ,w , n_c)
    pred_probs = model.predict(test_img)
    return np.argmax(pred_probs)


def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)