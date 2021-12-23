import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from photoitem import PhotoItem


class AI:
  def __init__(self, database_path, batch_size, epochs, initLoad, weights_filename):
    self.__database_path = database_path
    self.__batch_size = batch_size
    self.generate_labels()
    if initLoad: 
        self.load_trains()
        self.trains_logging()
    self.compile_model()
    self.model_logging()
    if initLoad: 
        self.fit(epochs)
        self.save_weights(weights_filename)
    else:
        self.load_weights(weights_filename)
  
  def generate_labels(self):
    future_labels = []
    self.labels = {}
    self.labels_length = 0

    for maybedir in os.listdir(path=self.__database_path):
      future_labels.append(maybedir)
    for key in future_labels:
      self.labels[key] = [0 for _ in range(len(future_labels))]
    for i in range(len(self.labels.keys())):  
      key = future_labels[i]
      self.labels[key][i] = 1
      self.labels_length += 1
    del future_labels
    print("Labels generated")

  
  def load_trains(self):
    self.x_train = []
    self.y_train = []

    self.__image_list = []
    for label in self.labels.keys():
      dir = self.__database_path + "/" + label
      flist = os.listdir(dir)
      for f in flist:
        fpath = self.__database_path + "/" + label + "/" + f
        self.__image_list.append(fpath)
    random.shuffle(self.__image_list)

    for fpath in self.__image_list:
      img = Image.open(fpath)
      img = self._smart_trimming(img)
      img = np.array(img)
      self.x_train.append(img)
      label = fpath.split('/')[-1].split('.')[0]
      self.y_train.append(self.labels[label])

    self.x_train = np.array(self.x_train)
    self.y_train = np.array(self.y_train)

    self.x_val = self.x_train[0:7000]
    self.y_val = self.y_train[0:7000]

    self.x_test = self.x_train[7000:10000]
    self.y_test = self.y_train[7000:10000]

    self.x_train = self.x_train[10000:]
    self.y_train = self.y_train[10000:]
    print("Trains generated")



  def trains_logging(self):
    print("Обучающая выборка:")
    print(self.x_train.shape)
    print(self.y_train.shape)

    print("Проверочная выборка:")
    print(self.x_val.shape)
    print(self.y_val.shape)

    print("Тестовая выборка:")
    print(self.x_test.shape)
    print(self.y_test.shape)
  

  def _smart_trimming(self, img):
    img_w, img_h = img.size
    target_size = [50, 50]
    if self._format_is_album(img_w, img_h):
      new_h = target_size[1]
      new_w = round(new_h / img_h * img_w)
    else:
      new_w = target_size[0]
      new_h = round(new_w / img_w * img_h)
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    center = [new_w//2, new_h//2]
    top_left = [center[0] - target_size[0]//2, center[1] - target_size[1]//2]
    bottom_right = [center[0] + target_size[0]//2, center[1] + target_size[1]//2]
    img = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    return img

  def _format_is_album(self, width: int, height: int):
    return True if width > height else False
  
  def compile_model(self, learning_rate=0.01):
    self.model = Sequential()
    self.model.add(BatchNormalization(input_shape=(50, 50, 3), name="bn1"))
    self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name="Conv2D-layer1"))
    self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name="Conv2D-layer2"))
    self.model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer1"))
    self.model.add(Dropout(0.25, name="Dropout-layer1"))
    self.model.add(BatchNormalization(name = "bn2"))
    self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name = "Conv2D-layer3"))
    self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name = "Conv2D-layer4"))
    self.model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer2"))
    self.model.add(Dropout(0.25, name="Dropout-layer2"))
    self.model.add(BatchNormalization(name = "bn3"))
    self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name = "Conv2D-layer5"))
    self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name = "Conv2D-layer6"))
    self.model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer3"))
    self.model.add(Dropout(0.25, name="Dropout-layer3"))
    self.model.add(Flatten(name="flatten"))
    self.model.add(Dense(len(self.labels.keys()), activation='softmax', name = "labels"))
    # self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    self.model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    print("Model compiled")

  def model_logging(self):
    self.model.summary()

  def fit(self, epochs=20):
    self.__history = self.model.fit(self.x_train, 
                            self.y_train, 
                            batch_size=self.__batch_size, 
                            epochs=epochs,
                            validation_data=(self.x_val,self.y_val),
                            verbose=1, 
                            use_multiprocessing=True,
                            )
    print("Model fitted")


  

  def fit_logging(self, n: int):
    plt.plot(self.__history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
    plt.plot(self.__history.history['val_accuracy'], 
            label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

    prediction = self.model.predict(self.x_test)

    print("Выход сети:")
    print(prediction[n])
    print()
    for i in range(2):
      print(i,"->","{:.40f}".format(prediction[n][i]))
    a = np.argmax(prediction[n])
      
    print("Распознан объект: ", a, "-", list(self.labels.keys())[a])
    print("Верный ответ: ", a, "-", list(self.labels.keys())[a])
  
  def save_weights(self,filepath):
    self.model.save_weights(filepath=filepath)
    print(f"Weights {filepath} saved")
  
  def load_weights(self, filepath):
    self.model.load_weights(filepath=filepath)
    print(f"Weights {filepath} loaded")

  def recognize_image(self, image: PhotoItem):
    img = Image.open(image.saved_path)
    img = self._smart_trimming(img)
    img = np.array(img).reshape((-1, 50, 50, 3))
    predictions = self.model.predict(
      img,
      batch_size=self.__batch_size, 
      verbose=1, 
      use_multiprocessing=True)
    labels = self.labels.copy()
    image.finish(*predictions[0])

