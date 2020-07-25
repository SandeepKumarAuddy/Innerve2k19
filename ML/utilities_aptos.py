import numpy as np
import PIL
import matplotlib.pyplot as plt
import requests
# from io import BytesIO
import io
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


# def save_image(url):
#     response = requests.get(url)
#     img = PIL.Image.open( io.BytesIO( response.content ) )
#     plt.imsave('img.png', np.array(img))

def save_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("img.png", 'wb') as f:
            f.write(response.content)


def hooked_backward_aptos(m, xb):
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            print(preds[0])
            print(preds)
            preds[0].backward()
    return hook_a, hook_g


def hooked_backward_brainy(m, xb):
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            # print(preds[2])
            print(preds[0][1])
            preds[0][1].backward()
    return hook_a, hook_g


def get_heatmap(learn, image, type):
    m = learn.model.eval();
    xb, _ = learn.data.one_item(image)
    xb_im = Image(learn.data.denorm(xb)[0])
    if (type == 'aptos'):
        hook_a, hook_g = hooked_backward_aptos(m, xb)
    elif (type == 'brainy'):
        hook_a, hook_g = hooked_backward_brainy(m, xb)
    acts = hook_a.stored[0].cpu()
    avg_acts = acts.mean(0)
    _, ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(avg_acts, alpha=0.3, extent=(0, 224, 224, 0),
              interpolation='bilinear');
    plt.savefig('heatmap/output_heat_map.png', bbox_inches='tight', pad_inches=0)
