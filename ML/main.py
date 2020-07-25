from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import numpy as np
from flask_restful import reqparse, abort, Api, Resource
from keras.models import load_model
from utilities_digit_recognize import predict_digit, get_image
import fastai
from fastai.vision import *
import timm
import os
from utilities_aptos import save_image, get_heatmap
from utilities_skin import *
from flask_cors import CORS

from rake_nltk import Rake

app = Flask(__name__, static_url_path='/heatmap')
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CLIENT_IMAGES"] = "/home/rishabh/PycharmProjects/Medidoc/heatmap"

learn_brainy: object = load_learner('model_brainy')
print('brainy loaded')
learn_aptos = load_learner('model').to_fp32()
print('aptos loaded')
learn_skinny = load_learner('model_skin')
print('skinny loaded')
learn_breast = load_learner('model_breast')
print('breast loaded')
learn_lungs = load_learner('model_lungs')
print('Lungs loaded')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/digit/", methods=['POST'])
def predict():
    model = load_model('mnist_model.h5')

    if request.method == 'POST':
        digit_name = request.get_json()
        comment = digit_name['url']
        img = get_image(comment)
        print(comment)
        ans = predict_digit(img, model)

    # return render_template('results.html', prediction=my_prediction, comment=comment)
    return jsonify({'result': int(ans)})


@app.route("/aptos/", methods=['POST'])
def predict_eyes():
    # load model
    # learn = load_learner('model').to_fp32()

    if request.method == 'POST':
        eye_name = request.get_json()
        image_url = eye_name['url']
        print(image_url)
        save_image(image_url)
        ans = learn_aptos.predict(open_image("img.png"))
        print(ans)
        ans = ans[2].item()
        my_prediction = int((ans / 4) * 100)
        print(my_prediction)

        img = open_image("img.png")
        get_heatmap(learn_aptos, img, type='aptos')
        os.remove("img.png")
        url_hm = url_for('static', filename='output_heat_map.png')
        print(url_hm)

    return jsonify({'result': my_prediction})


@app.route("/brainy/", methods=['POST'])
def predict_tumour():
    # load model
    # learn = load_learner('model_brainy')

    if request.method == 'POST':
        tumour_name = request.get_json()
        image_url = tumour_name['url']
        print(image_url)
        save_image(image_url)
        ans = learn_brainy.predict(open_image("img.png"))
        ans = ans[2][1].item()
        my_prediction = int(ans * 100)
        print(my_prediction)

        # img = open_image("img.png")
        # get_heatmap (learn_brainy, img, type='brainy')
        # os.remove("img.png")
        # url_hm = url_for('static', filename='output_heat_map.png')
        # print(url_hm)

    return jsonify({'result': my_prediction})


@app.route("/get-image/<image_name>")
def get_image(image_name):
    try:
        return send_from_directory(app.config["CLIENT_IMAGES"], filename=image_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)


@app.route("/skinny/", methods=['POST'])
def predict_skin_cancer():
    if request.method == 'POST':
        skin_name = request.get_json()
        image_url = skin_name['url']
        print(image_url)
        save_image(image_url)

        ans = learn_skinny.predict(open_image(f'img.png'))
        print(ans)
        ans = ans[1].item()
        my_prediction = int(ans)
        print(my_prediction)

    return jsonify({'result': my_prediction})


@app.route("/keyword/", methods=['POST'])
def predict_digits():
    if request.method == 'POST':
        transcript = request.get_json()
        script = transcript['data']
        r = Rake()
        r.extract_keywords_from_text(script)
        predicted_ans = ''.join(r.get_ranked_phrases())

    return jsonify({'result': predicted_ans})


@app.route("/breast/", methods=['POST'])
def predict_breast():
    if request.method == 'POST':
        breast_img = request.get_json()
        image_url = breast_img['url']
        print(image_url)
        save_image(image_url)

        ans = learn_breast.predict(open_image("img.png"))
        print(ans)
        ans = ans[1].item()
        my_prediction = int(ans)
        print(my_prediction)

    # the ans will be of form 0 ( Normal ), 1 ( Benign ), 2 ( Malignant )

    return jsonify({'result': my_prediction})


@app.route("/pneumonia/", methods=['POST'])
def predict_pneumonia():
    if request.method == 'POST':
        lung_image = request.get_json()
        image_url = lung_image['url']
        print(image_url)
        save_image(image_url)

        ans = learn_lungs.predict(open_image("img.png"))
        print(ans)
        ans = ans[2][1].item() * 100
        my_prediction = int(ans)
        print(my_prediction)

        # ans will be of form 0 or 1

    return jsonify({'result': my_prediction})


if __name__ == '__main__':
    app.run(host="10.10.1.224", port=5000, debug=True)
