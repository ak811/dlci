# Download the dataset from:
# https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# 12,500 images of dogs and 12,500 images of cats

import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.preprocessing import image

# generating images
image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               rescale=1 / 255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

# organize the images in sub-directories
image_gen.flow_from_directory('../data/train')
image_gen.flow_from_directory('../data/test')

# resizing the images
image_shape = (150, 150, 3)

# building the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

# reduce overfitting by 50%
model.add(Dropout(0.5))

# cat: 0, dog: 1
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# model training
batch_size = 16

train_image_gen = image_gen.flow_from_directory('../data/train',
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory('../data/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

results = model.fit_generator(train_image_gen, epochs=100,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                              validation_steps=12)

# model.save('my_model.h5')
plt.plot(results.history['acc'])

# predict new images
dog_file = '../data/train/dog/1.jpg'
dog_image = image.load_img(dog_file, target_size=(150, 150))
dog_image = image.img_to_array(dog_image)
dog_image = np.expand_dims(dog_image, axis=0)
dog_image = dog_image / 255

prediction_prob = model.predict(dog_image)
print('dog probability: ' + prediction_prob)
