# coding: utf-8
import numpy as np
import os, time, pickle
import keras
from keras import layers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint


# 1.transfer learning - reconstruct net
base_model = VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))

x = base_model.output
x = layers.Flatten()(x)
#x = layers.Dense(1024, activation='relu')(x)
# 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
#x = layers.Dropout(0.5)(x)
#x = layers.Dense(512, activation='relu')(x)
#out = layers.Dropout(0.3)(out)
preds = layers.Dense(193, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

# for layer in model.layers[:20]:
#     layer.trainable = False
# for layer in model.layers[20:]:
#     layer.trainable = True

# # view model info
# model.summary()
# for i,layer in enumerate(model.layers):
#   print(i,layer.name)
# exit()

# 2.preprocess data to prepare dataset
train_datagen = ImageDataGenerator(#horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory('../input/data/train/',
                                                    target_size=(128, 128),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)
                                                    #save_to_dir=r'./trainresult')

val_datagen = ImageDataGenerator(#horizontal_flip=True,
    preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_directory('../input/data/validation/',
                                                    target_size=(128, 128),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

classes = train_generator.class_indices
with open("./correspondence.pickle", 'wb') as f:
    pickle.dump(classes, f)

# 3.set hyperparameters
sgd = optimizers.SGD(lr=0.001, momentum=0.9)  # decay=1e-6, nesterov=True
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4.train
step_size_train = train_generator.n  # train_generator.batch_size
filepath = 'model-ep{epoch:03d}-acc{acc:.3f}.h5'
checkpoint = ModelCheckpoint(
    filepath, monitor='acc', verbose=1, mode='max')
callback_lists = [TensorBoard(log_dir="./log"), checkpoint]

start_t = time.time()
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=50,
                    callbacks=callback_lists,
                    validation_data=val_generator,
                    validation_steps=step_size_train)
end_t = time.time()
m, s = divmod(end_t-start_t, 60)
print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s to train the new!\n")
