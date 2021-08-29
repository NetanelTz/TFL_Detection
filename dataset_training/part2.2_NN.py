"""
Authors: Matan Omesi,
         Omer Hadad,
         Netanel Tzadok,
         Ariel Haser,
         Yaron Gerstel
"""

SHAPE = (81, 81, 3)
SIZE_CROP = 41
FILES_USE_COUNT = False  # true means that building dataset from FILE_COUNT amount of files
FILES_START_INDEX = 0   # start scan files from position INDEX with limit of COUNT
FILES_COUNT = 1000
TFL_SIGN = b'1'
NOT_TFL_SIGN = b'0'

# files
DATA_DIR_PATH = r'C:\networks\work\mobileye'
DATA_BIN_FILE_NAME = r'../data.bin'
LABELS_BIN_FILE_NAME = r'../labels.bin'
JSON_PATH = r'../gtFine/train/*/*polygons.json'
TRAIN_PATH_DIRECTORY = r'../leftImg8bit_trainvaltest/leftImg8bit/train/'
TRAIN_DIR_NAME = 'train'
VAL_DIR_NAME = 'val'


# replace name file extension
FILE_EXT_FROM = 'gtFine_polygons.json'
FILE_EXT_TO = 'leftImg8bit.png'


# Train File
LABELS_TO_NAME = {0: 'No TFL', 1: 'Yes TFL'}

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import seaborn as sbn




def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, DATA_BIN_FILE_NAME),
                       mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, LABELS_BIN_FILE_NAME), mode='r', dtype=np.uint8)
    labels = np.array(list(map(lambda x: x - 48, labels)))

    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name=LABELS_TO_NAME):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05},
                      squeeze=False, sharex=True, sharey=True)[1]  # .flatten()
    index_shape = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(index_shape):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None:
            title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)

    plt.show()


# root = './'  #this is the root for your val and train datasets
datasets = {
    'val': load_tfl_data(join(DATA_DIR_PATH, VAL_DIR_NAME)),
    'train': load_tfl_data(join(DATA_DIR_PATH, TRAIN_DIR_NAME)),
}

for k, v in datasets.items():
    print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

viz_my_data(num=(6, 6), **datasets['val'])


# ####################### define the model used for training ###################


def tfl_model():
    input_shape = (81, 81, 3)

    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for i in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(16, kernel_size=(3, 3), input_shape=input_shape)
    spatial_layer(1, 8)
    spatial_layer(2, 8)
    spatial_layer(2, 8)
    #spatial_layer(3, 8)
    # spatial_layer(2, 8)

    model.add(Flatten())
    dense_bn_relu(16)
    model.add(Dense(3, activation='softmax'))
    return model


m = tfl_model()
m.summary()

# ############################ train #####################

datasets = {
    'val': load_tfl_data(join(DATA_DIR_PATH, VAL_DIR_NAME)),
    'train': load_tfl_data(join(DATA_DIR_PATH, TRAIN_DIR_NAME)),
}

# prepare our model
m = tfl_model()
m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])

train, val = datasets['train'], datasets['val']

# train it, the model uses the 'train' dataset for learning.
# We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
history = m.fit(train['images'], train['labels'],
                validation_data=(val['images'], val['labels']), epochs=8)

# compare train vs val accuracy,
# why is val_accuracy not as good as train accuracy? are we over fitting?


# compare train vs val accuracy,
# why is val_accuracy not as good as train accuracy? are we over fitting?
epochs = history.history
epochs['train_acc'] = epochs['accuracy']

plt.figure(figsize=(10, 10))
for k in ['train_acc', 'val_accuracy']:
    plt.plot(range(len(epochs[k])), epochs[k], label=k)
plt.show()
plt.legend()

# ##################### evaluate and predict ####################

predictions = m.predict(val['images'])
sbn.distplot(predictions[:, 0])

predicted_label = np.argmax(predictions, axis=-1)
print('accuracy:', np.mean(predicted_label == val['labels']))

viz_my_data(num=(6, 6), predictions=predictions[:, 1], **val)

# ############# save the model ##################

m.save("model.h5_2")