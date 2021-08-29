import RGB_filtering
from dataset_training.part2_1_cut_picture import send_images_from_array
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

PATH_PLS = r'./play_list.pls'
f = open(PATH_PLS, 'r')
lst_pls_path = []
for path in f:
    lst_pls_path.append(path)
f.close()
PKL_PATH = lst_pls_path[0]
IMAGE_NUM = lst_pls_path[1]
LST_PATH_IMG = lst_pls_path[2:]


class TFL_main:

    def __init__(self, path_img: str, img_num: int):
        self._path = path_img
        self._num = img_num
        self._lst_candidate = RGB_filtering.main(self._path)

    def save_and_crop_img(self):
        return send_images_from_array(np.array(Image.open(self._path)), self._lst_candidate)


def controler():
    for i, path in enumerate(LST_PATH_IMG):
        f = TFL_main(path[:-1], eval(IMAGE_NUM) + i)
        array_img, pixels_array = f.save_and_crop_img()
        loaded_model = load_model("model.h5")
        accuracy = []
        for i, img in enumerate(array_img):
            accuracy.append([loaded_model.predict(img)[0][1], pixels_array[i]])

        print(accuracy)


if __name__ == '__main__':
    controler()
