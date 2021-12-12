import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5")
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


labels = {
    "dog": [1, 0],
    "cat": [0, 1],
}

dataset_path = "."


class AI:
  def __init__(self, database_path, batch_size):
    self.database_path = database_path
    self.batch_size = batch_size
    self.__generate_labels()
    self.__load_image_list()
    self.__init_trains()
    self.__load_image()
    self.__xy_logging()
    print('AI has been initialized')
  
  def __generate_labels(self):
    future_labels = []
    self.labels = {}
    for maybedir in os.listdir(path=self.database_path):
      future_labels.append(maybedir)
    for key in future_labels:
      self.labels[key] = [0 for _ in range(len(future_labels))]
    for i in range(len(self.labels.keys())):  
      key = future_labels[i]
      self.labels[key][i] = 1
    del future_labels


  def __load_image_list(self):
    self.image_list = []
    for label in self.labels.keys():
      dir = self.database_path + "/" + label
      flist = os.listdir(dir)
      for f in flist:
        fpath = self.database_path + "/" + label + "/" + f
        self.image_list.append(fpath)
    random.shuffle(self.image_list)

  def __init_trains(self):
    self.x_train = []
    self.y_train = []
    self.x_val = []
    self.y_val = []
    self.x_test = []
    self.y_test = []
  
  def __convert_trains_to_ndarray(self):
    self.x_train = np.ndarray(self.x_train)
    self.y_train = np.ndarray(self.y_train)

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

  def __load_image(self):
    for fpath in self.image_list:
      img = Image.open(fpath)
      img = self.__smart_trimming(img)
      img = np.array(img)
      self.x_train.append(img)
      label = fpath.split('/')[-1].split('.')[0]
      self.y_train.append(self.labels[label])
      print(fpath.split('/')[-1].split('.')[1])

    self.x_val.append(self.x_train[0:7000])
    self.y_val.append(self.y_train[0:7000])

    self.x_test.append(self.x_train[7000:10000])
    self.y_test.append(self.y_train[7000:10000])

    self.x_train.append(self.x_train[10000:])
    self.y_train.append(self.y_train[10000:])
  

  def __xy_logging(self):
    print("Обучающая выборка:")
    print(self.x_train.shape)
    print(self.y_train.shape)

    print("Проверочная выборка:")
    print(self.x_val.shape)
    print(self.y_val.shape)

    print("Тестовая выборка:")
    print(self.x_test.shape)
    print(self.y_test.shape)


Ai = AI(database_path="img", batch_size=128)
exit(0)
# model = Sequential()
# model.add(BatchNormalization(input_shape=(120, 120, 3), name="bn1"))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name="Conv2D-layer1"))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name="Conv2D-layer2"))
# model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer1"))
# model.add(Dropout(0.25, name="Dropout-layer1"))
# model.add(BatchNormalization(name = "bn2"))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name = "Conv2D-layer3"))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name = "Conv2D-layer4"))
# model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer2"))
# model.add(Dropout(0.25, name="Dropout-layer2"))
# model.add(BatchNormalization(name = "bn3"))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name = "Conv2D-layer5"))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name = "Conv2D-layer6"))
# model.add(MaxPooling2D(pool_size=(2, 2), name="mp2D-layer3"))
# model.add(Dropout(0.25, name="Dropout-layer3"))
# model.add(Flatten(name="flatten"))
# model.add(Dense(2, activation='softmax', name = "labels"))
# #model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

# model.summary()

# history = model.fit(x_train, 
#                     y_train, 
#                     batch_size=batch_size, 
#                     epochs=20,
#                     validation_data=(x_val, y_val),
#                     verbose=1)

# plt.plot(history.history['accuracy'], 
#          label='Доля верных ответов на обучающем наборе')
# plt.plot(history.history['val_accuracy'], 
#          label='Доля верных ответов на проверочном наборе')
# plt.xlabel('Эпоха обучения')
# plt.ylabel('Доля верных ответов')
# plt.legend()
# plt.show()

# prediction = model.predict(x_test)
# n = 123 # 1022 # 1137 # 1132 #1122

# img = x_test[n]
# img = img.reshape(120, 120, 3)
# img = img.astype('uint8')
# plt.figure(figsize=(2, 2))
# plt.imshow(Image.fromarray(img))
# plt.show()

# print("Выход сети:")
# print(prediction[n])
# print()
# for i in range(2):
#   print(i,"->","{:.40f}".format(prediction[n][i]))
# a = np.argmax(prediction[n])
# if a == 0:
#   b = 'dog'
# else:
#   b = 'cat'
# print("Распознан объект: ", a, "-", b)
# print("Верный ответ: ", a, "-", b)