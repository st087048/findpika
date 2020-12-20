import numpy as np
import pandas as pd
import os
import random
import gc
import cv2
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

def read_and_process_image(list_of_images):
    global nrows
    global ncolumns
    x = []
    y = []
    i = 0
    for image in list_of_images:

        x.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        if len(x) == i + 1:
            if 'pikachu' in image:
                y.append(1)
            elif 'not_pika' in image:
                y.append(0)
        i += 1

    return x,y


os.chdir("C:/Program Files (x86)/123")

train_dir = 'C:\\Users\\T-PC\\PycharmProjects\\pythonProject3\\Train'
test_dir = 'C:\\Users\\T-PC\\PycharmProjects\\pythonProject3\\Test'

train_pikachu = ['C:/Users/T-PC/PycharmProjects/pythonProject3/Train/{}'.format(i) for i in os.listdir(train_dir) if 'pikachu' in i]
train_not_pika = ['C:/Users/T-PC/PycharmProjects/pythonProject3/Train/{}'.format(i) for i in os.listdir(train_dir) if 'not_pika' in i]


test_imgs = ['C:/Users/T-PC/PycharmProjects/pythonProject3/Test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_pikachu + train_not_pika
random.shuffle(train_imgs)
del train_pikachu
del train_not_pika

gc.collect()



nrows = 150
ncolumns = 150
channels = 3

x, y = read_and_process_image(train_imgs)
plt.figure(figsize=(20, 10))
columns = 5
#for i in range(columns):
 #   plt.subplot(5 / columns + 1, columns, i + 1)
  #  plt.imshow(x[i])
#plt.show()
del train_imgs
gc.collect()

x = np.array(x)
y = np.array(y)
print(len(x))
print(len(y))
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=2)
del x
del y
gc.collect()

ntrain = len(x_train)
nval = len(x_val)

batch_size = 32

####

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

####

model.summary()

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs= 2,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)

model.save_weights('weights1.ckpt')
model.save('model.ckpt')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
mas = []
for i in range(9):
    ind = random.randint(0,len(test_imgs)-1)
    mas.append(test_imgs[ind])

mas = mas + ['C:\\Users\\T-PC\\PycharmProjects\\pythonProject3\\Test\\536633.700xp.jpg']
X_test, y_test = read_and_process_image(mas) #Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)
i = 0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('pikachu')
    else:
        text_labels.append('not pikachu')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()