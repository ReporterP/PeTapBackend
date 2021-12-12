import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1")
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adadelta 
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import math


class AI:
  def __init__(self, database_path, batch_size):
    self.__database_path = database_path
    self.__batch_size = batch_size
    self.__generate_labels()
    self.__load_image_list()
    self.__init_trains()
    self.__load_images()
    self.__xy_logging()
    self.__compile_model()
    self.__log_model()
    self.__studying_ai()
    self.__studying_log()
  
  def __generate_labels(self):
    future_labels = []
    self.__labels = {}
    for maybedir in os.listdir(path=self.__database_path):
      future_labels.append(maybedir)
    for key in future_labels:
      self.__labels[key] = [0 for _ in range(len(future_labels))]
    for i in range(len(self.__labels.keys())):  
      key = future_labels[i]
      self.__labels[key][i] = 1
    del future_labels

  def __load_image_list(self):
    self.__image_list = []
    for label in self.__labels.keys():
      dir = self.__database_path + "/" + label
      flist = os.listdir(dir)
      for f in flist:
        fpath = self.__database_path + "/" + label + "/" + f
        self.__image_list.append(fpath)
    random.shuffle(self.__image_list)

  def __init_trains(self):
    self.__x_train = []
    self.__y_train = []
  
  def __smart_trimming(self, img):
    img_w, img_h = img.size
    target_size = [120, 120]
    if self.__format_is_album(img_w, img_h):
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

  def __format_is_album(self, width: int, height: int):
    return True if width > height else False

  def __load_images(self):
    for fpath in self.__image_list:
      img = Image.open(fpath)
      img = self.__smart_trimming(img)
      img = np.array(img)
      self.__x_train.append(img)
      label = fpath.split('/')[-1].split('.')[0]
      self.__y_train.append(self.__labels[label])

    self.__x_train = np.array(self.__x_train)
    self.__y_train = np.array(self.__y_train)

    self.__x_val = self.__x_train[0:7000]
    self.__y_val = self.__y_train[0:7000]

    self.__x_test = self.__x_train[7000:10000]
    self.__y_test = self.__y_train[7000:10000]

    self.__x_train = self.__x_train[10000:]
    self.__y_train = self.__y_train[10000:]
  

  def __xy_logging(self):
    print("Обучающая выборка:")
    print(self.__x_train.shape)
    print(self.__y_train.shape)

    print("Проверочная выборка:")
    print(self.__x_val.shape)
    print(self.__y_val.shape)

    print("Тестовая выборка:")
    print(self.__x_test.shape)
    print(self.__y_test.shape)

  def __compile_model(self):
    self.__model = Sequential()
    self.__model.add(BatchNormalization(input_shape=(120, 120, 3), name="bn1"))
    self.__model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name="Conv2D-layer1"))
    self.__model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name="Conv2D-layer2"))
    self.__model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer1"))
    self.__model.add(Dropout(0.25, name="Dropout-layer1"))
    self.__model.add(BatchNormalization(name = "bn2"))
    self.__model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name = "Conv2D-layer3"))
    self.__model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name = "Conv2D-layer4"))
    self.__model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer2"))
    self.__model.add(Dropout(0.25, name="Dropout-layer2"))
    self.__model.add(BatchNormalization(name = "bn3"))
    self.__model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name = "Conv2D-layer5"))
    self.__model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name = "Conv2D-layer6"))
    self.__model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer3"))
    self.__model.add(Dropout(0.25, name="Dropout-layer3"))
    self.__model.add(Flatten(name="flatten"))
    self.__model.add(Dense(len(self.__labels.keys()), activation='softmax', name = "labels"))
    #model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
    self.__model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

  def __log_model(self):
    self.__model.summary()

  def __studying_ai(self):
    self.__history = self.__model.fit(self.__x_train, 
                            self.__y_train, 
                            batch_size=self.__batch_size, 
                            epochs=20,
                            validation_data=(self.__x_val,self.__y_val),
                            verbose=1)

  def __studying_log(self):
    plt.plot(self.__history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
    plt.plot(self.__history.history['val_accuracy'], 
            label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()

    prediction = self.__model.predict(self.__x_test)
    n = 123

    img = self.__x_test[n]
    img = img.reshape(120, 120, 3)
    img = img.astype('uint8')
    plt.figure(figsize=(2, 2))
    plt.imshow(Image.fromarray(img))
    plt.show()

    print("Выход сети:")
    print(prediction[n])
    print()
    for i in range(2):
      print(i,"->","{:.40f}".format(prediction[n][i]))
    a = np.argmax(prediction[n])
    if a == 0:
      b = 'dog'
    else:
      b = 'cat'
    print("Распознан объект: ", a, "-", b)
    print("Верный ответ: ", a, "-", b)
  
sa = AI(database_path="img", batch_size=128)