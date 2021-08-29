import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sbn
from tensorflow.keras.models import load_model

loaded_model = load_model("model.h5")
SHAPE = (81, 81, 3)
SIZE_CROP = 41
FILES_USE_COUNT = False  # true means that building dataset from FILE_COUNT amount of files
FILES_START_INDEX = 0  # start scan files from position INDEX with limit of COUNT
FILES_COUNT = 1000
TFL_SIGN = b'1'
NOT_TFL_SIGN = b'0'

# files
DATA_DIR_PATH = r'/'
DATA_BIN_FILE_NAME = r'../data.bin'
LABELS_BIN_FILE_NAME = r'../labels.bin'
JSON_PATH = r'../gtFine/train/*/*polygons.json'
TRAIN_PATH_DIRECTORY = r'../leftImg8bit_trainvaltest/leftImg8bit/train/'
TEST_DIR_NAME = ''



# Train File
LABELS_TO_NAME = {0: 'No TFL', 1: 'Yes TFL'}

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
        if predictions[idx] >= 0.6: # threshold = 0.6 because it's T.L..
            predictions[idx] = 1
        if predictions is not None:
            title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)

    plt.show()


datasets = {
    'test': load_tfl_data(DATA_DIR_PATH)
}
val = datasets['test']


loaded_model = load_model("model.h5")

# code copied from the training evaluation:
l_predictions = loaded_model.predict(val['images'])
sbn.distplot(l_predictions[:,0])
l_predicted_label = np.argmax(l_predictions, axis=-1)
viz_my_data(num=(6, 6), predictions=l_predictions[:, 1], **val)

print ('accuracy:', np.mean(l_predicted_label==val['labels']))