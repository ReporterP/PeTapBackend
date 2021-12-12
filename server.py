from PIL import Image
from flask import Flask, request, abort, send_file, jsonify
from flask.json import load
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import os

import artifical_intellegence
import database
from photoitem import PhotoItem

load_images_dir = "downloads"
os.system('rm -rf ' + load_images_dir + '/*')

db = database.Database()
# ai = artifical_intellegence.AI(database_path="img", batch_size=128)

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    filename = secure_filename(f.filename)
    saved_path = load_images_dir + "/" + filename
    f.save(saved_path)
    img = Image.open(saved_path)
    key = db.add(PhotoItem(img, saved_path))
    return jsonify({"key": key})


@app.route(f'/photo/<key>', methods=['GET'])
def photo(key):
    selectImage = db.get(key)
    return jsonify(selectImage.toMap())

@app.route(f'/photo/<key>/show', methods=['GET'])
def photo_show(key):
    selectImage = db.get(key)
    return send_file(selectImage.saved_path, mimetype='image/gif')

# TODO: Найти датасет из +- 1000 фоток кошбкособак
# TODO: Пройтись по всем фоткам и разобрать ответ нейросети от них
# TODO: Записать ответ в отдельный файлы
# TODO: Реализовать методы игры

# TODO: Добавить работу нейросети в существующую нейронку
app.run(debug=True, port=3333)

