from PIL import Image
from flask import Flask, request, abort, send_file, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

import artifical_intellegence
import database

load_images_dir = "downloads"

db = database.Database()
# ai = artifical_intellegence.AI()
# sa = AI(database_path="img", batch_size=128)

app = Flask(__name__)


# Метод загрузки фотографии, где возвращается id в базе
# Информация по id
# Список последних (рандомных) фоток

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    img = Image.open(f.filename)
    plt.figure()
    plt.imshow(img)
    plt.show()
    # Текущая фотография - filename
    return 'Ok'

# Выводит последнюю подгруженную фотографию

@app.route(f'/show/<id>', methods=['GET'])
def show(id):
    print(id)
    # Смотрим результат по id
    # return send_file(listFilename[len(listFilename)-1], mimetype='image/gif')

app.run(debug=True, port=3333)
print('Server run')