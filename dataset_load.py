import numpy as np
import PIL
from PIL import Image
import os
import pandas as pd

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
print('Reading in data...')
train_images = []
train_files = train_files[0:10]
for f in train_files:
  img = Image.open(train_path + f)
  train_Y.append(labels[labels['file'] == f]['rating'])
  #img = img.resize(image_size)
  img_arr = np.array(img)
  train_images.append(img_arr)

train_X = np.array(train_images)
train_Y = np.array(train_Y)[:,0]
print('Shape of Train Set:')
print(np.shape(train_X))

print('Shape of Train Labels:')
print(np.shape(train_Y))
print(train_Y)

# print(train_X[1])
# print(np.shape(train_X))