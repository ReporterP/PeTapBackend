from flask import Flask, request, abort, send_file, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

database = {}

# Документация на /
# Метод загрузки фотографии, где возвращается id в базе
# Информация по id
# Список последних (рандомных) фоток


@app.route('/')
def index():
    # Документация
    # return jsonify(listFilename)

# Позволяет загрузить фотографию от пользователя на сервер.
# Создается запрос с form-data, где по ключу "the_file" передаётся файл.
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)
    # Текущая фотография - filename
    return ''

# Выводит последнюю подгруженную фотографию

@app.route(f'/show/<id>', methods=['GET'])
def show(id):
    print(id)
    # Смотрим результат по id
    # return send_file(listFilename[len(listFilename)-1], mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)