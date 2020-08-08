import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
import os
import pandas as pd
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Concatenate, AvgPool2D, Dropout, Activation
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras import Model, Input



label_path = './reliability_ratings2.csv'
train_path = '/data/ukbb/fundus_image_grading/images_for_reliability_check/'
left_path = '/data/ukbb/ukbb_fundus_lefteye_images/21015/'
right_path = '/data/ukbb/ukbb_fundus_righteye_images/21016/'

labels = pd.read_csv(label_path)
labels['file'] = labels['file'].apply(lambda x: x.split(sep = '/')[-1])

left_files = os.listdir(left_path)
right_files = os.listdir(right_path)
train_files = os.listdir(train_path)

left_images = [y for y in left_files if '.png' in y]
right_images = [y for y in right_files if '.png' in y]

train_Y = []

#image_size = (2048,1536)
print('\n\n\nReading in data...')
train_images = []
train_files = train_files[0:15]
for f in train_files:
  img = Image.open(train_path + f)
  train_Y.append(labels[labels['file'] == f]['rating'])
  #img = img.resize(image_size)
  img_arr = np.array(img)
  train_images.append(img_arr)

train_X = np.array(train_images)
train_Y = np.array(train_Y)[:,0]
print('\n\n\nShape of Train Set:')
print(np.shape(train_X))

print('\n\n\nShape of Train Labels:')
print(np.shape(train_Y))
print(train_Y)


# Load the TensorBoard notebook extension (optional)
# %load_ext tensorboard

tf.random.set_seed(42)
tf.__version__

tf.config.list_physical_devices(device_type=None)
print("\n\n\nNum GPUs:", len(tf.config.list_physical_devices('GPU')))

print('\n\n\nSetting up Architecture')
INPUT_SHAPE = np.shape(train_X[0])
vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

print('\n\n\nCreating VGG Output')
x = vgg.layers[-1].output
x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(3, activation='sigmoid')(x)

print('\n\n\nCompiling Model:')
vgg = Model(vgg.input, x)
vgg.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# # Freeze the layers
# for layer in vgg.layers[:5]:
#     layer.trainable = False

print('\n\n\nTraining Model...')
vgg.fit(train_X[0:10], train_Y[0:10], validation_data=(train_X[10:15],train_Y[10:15]), epochs=10, batch_size=64)


