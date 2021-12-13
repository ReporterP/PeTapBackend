from flask import Flask, request, abort, send_file, jsonify
from werkzeug.utils import secure_filename
import os

import artifical_intellegence
import database
from photoitem import PhotoItem

load_images_dir = "downloads"
os.system('rm -rf ' + load_images_dir)
os.system('mkdir ' + load_images_dir)

db = database.Database()

ai = artifical_intellegence.AI(
                            database_path="dataset", 
                            batch_size=128, 
                            epochs=20, 
                            initLoad=False, 
                            weights_filename="dogandcats.hdf5")

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    filename = secure_filename(f.filename)
    saved_path = load_images_dir + "/" + filename
    f.save(saved_path)
    photo_item = PhotoItem(saved_path)
    key = db.add(photo_item)
    db.get(key)
    ai.recognize_image(photo_item)
    return jsonify({"key": key})


@app.route(f'/photo/<key>', methods=['GET'])
def photo(key):
    selectImage = db.get(key)
    return jsonify(selectImage.toMap())

@app.route(f'/photo/<key>/show', methods=['GET'])
def photo_show(key):
    selectImage = db.get(key)
    return send_file(selectImage.saved_path, mimetype='image/gif')

app.run(debug=False, port=3333)

