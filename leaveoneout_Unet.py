import numpy as np
import keras 
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization,Flatten, Dropout
from keras.layers import concatenate
from keras.optimizers import *
from keras.utils import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K
import math
import random
import matplotlib.pyplot as plt
#%matplotlib inline
import h5py

from scipy import misc, ndimage
import glob
import imageio
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.95 # ??GPU90%??? 
session = tf.Session(config=config)


def load_data(image_paths):
    n_samples = len(image_paths)
    images = np.zeros((n_samples, image_size, image_size, 1))
    for i in range(n_samples):
        img = imageio.imread(image_paths[i])
#         img= img.reshape(512,512,1)
        images[i,:,:,:] = np.expand_dims(img/255, axis = -1)
    return images

#load image masks into a database
def load_data_labels(image_paths):
    n_samples = len(image_paths)
    images = np.zeros((n_samples, image_size, image_size, 1))
    for i in range(n_samples):
        img = imageio.imread(image_paths[i])
        images[i,:,:,:] = np.expand_dims(img/255, axis = -1)
        
    return images

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return 1-((2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth))



image_size =992

imgs_train = load_data(glob.glob("deform/8_train/*.tif"))
imgs_mask_train = load_data_labels(glob.glob("deform/8_label/*.tif"))
imgs_test = load_data(glob.glob("deform/8_test/*.tif"))
imgs_mask_test=load_data_labels(glob.glob("deform/8_test_mark/*.tif"))



print("Train data", imgs_train.shape)
print("Train labels",imgs_mask_train.shape)
print("Test data",imgs_test.shape)
print("Test labels",imgs_mask_test.shape)


input = Input(shape=(image_size,image_size,1))
K.set_image_dim_ordering('tf')
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',dim_ordering="tf")(input)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',dim_ordering="tf")(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',dim_ordering="tf")(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',dim_ordering="tf")(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',dim_ordering="tf")(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',dim_ordering="tf")(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',dim_ordering="tf")(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',dim_ordering="tf")(conv4)
conv4 = BatchNormalization()(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',dim_ordering="tf")(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',dim_ordering="tf")(conv5)
conv5 = BatchNormalization()(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
conv6 = BatchNormalization()(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
conv7 = BatchNormalization()(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
conv8 = BatchNormalization()(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
conv9 = BatchNormalization()(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs = input, outputs = conv10)


model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics = ['accuracy',mean_iou])

model.summary()


#filepath = 'weights1.{epoch:02d}-{loss:.2f}.hdf5'
#ck = keras.callbacks.ModelCheckpoint(filepath,monitor='loss',verbose=0)

#G = 1
#for d in ('/gpu:4'):
#    with tf.device(d):
#        M = Unet_model()
#        
#model = keras.utils.multi_gpu_model(M, gpus=G)

history = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=350, verbose=1
                  )



#model.save_weights('28.h5')


#model.load_weights('UNet_992_8.h5')

print('\nfig8')
score=model.evaluate(imgs_test,imgs_mask_test,batch_size=1)
print('Test score = ',score)
mask_on_test = model.predict(imgs_test, batch_size=1, verbose=1)
print('\nprediction',mask_on_test)

plt.figure(figsize=(992/96,992/96), dpi = 96)
plt.imshow(mask_on_test[0].reshape(992, 992))
plt.savefig('8.tif')
plt.show()

