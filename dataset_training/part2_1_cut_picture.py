import glob
import json
from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

SIZE = 41


def crop_tfl_img(img_arr, ymax, ymin, xmax, xmin):
    cropped_img = np.pad(img_arr, ((40, 40), (40, 40), (0, 0)), 'constant', constant_values=0)
    xpad = (80 - (xmax - xmin)) // 2 + 1
    ypad = (80 - (ymax - ymin)) // 2 + 1
    cropped_img = cropped_img[xmin + 41 - xpad:xmax + 41 + xpad,
                  ymin + 40 - ypad:ymax + 40 + ypad]

    new_image = np.array(cropped_img)
    shape = cropped_img.shape
    if shape[0] > 81:
        new_image = cropped_img[:-(shape[0]-81), :]

    if shape[1] > 81:
        new_image = new_image[:, :-(shape[1] - 81)]

    #print(shape, new_image.shape)
    return new_image # np.reshape(cropped_img,(81, 81, 3))


def crop_notfl_img(img_arr, pixels):
    print(pixels)
    cropped_img = np.pad(img_arr, ((40, 40), (40, 40), (0, 0)), 'constant', constant_values=0)

    x, y = round(pixels[0]+40), round(pixels[1]+40)
    print(y,x)
    cropped_img = cropped_img[y - SIZE:y + SIZE, x - SIZE:x + SIZE]
    # plt.imshow(cropped_img)
    # plt.show()
    new_image = np.array(cropped_img)
    shape = cropped_img.shape
    if shape[0] > 81:
        new_image = cropped_img[:-(shape[0] - 81), :]
    if shape[1] > 81:
        new_image = new_image[:, :-(shape[1] - 81)]
    if pixels[0] == 930 and pixels[1] == 157: # [930, 157]:
        plt.imshow(new_image)
        plt.show()
    #print(shape, new_image.shape)
    return new_image
    # return np.reshape(cropped_img,(81, 81, 3))


# def get_lable_data(path_image, grey_image: np.ndarray):
#     # traffic lights is 19
#     all_tl = np.where(grey_image == 19)
#     print(all_tl)
#     if len(all_tl[0]) == 0:
#         return []
#     return all_tl


def open_json(path, image):
    with open(path) as j_file:
        j_obj = json.load(j_file)
        j_file.close()
    count = 0
    tfl_in_img = []
    for elem in j_obj['objects']:
        if elem['label'] == 'traffic light':
            ymax = max(elem['polygon'], key=lambda x: x[0])[0]
            ymin = min(elem['polygon'], key=lambda x: x[0])[0]
            xmax = max(elem['polygon'], key=lambda x: x[1])[1]
            xmin = min(elem['polygon'], key=lambda x: x[1])[1]
            tfl_in_img.append(crop_tfl_img(image, ymax, ymin, xmax, xmin))
            count += 1
    return tfl_in_img


# 010110101


def save_binary_file(data):
    with open('../data.bin', 'ab') as data_file, open('../labels.bin', 'ab') as labels_file:
        for item in data:
            cropped_img = item[0]
            status = eval(item[1])
            print('Writing to files: ', status, 'with shape: ', cropped_img.shape)
            data_file.write(np.array(cropped_img).tobytes())
            labels_file.write(status.to_bytes())
    print('Done')


def load_binary_data(data_dir='./', crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, '../data.bin'),
                       mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, '../labels.bin'),
                       mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}

# [1,2,3,4,5]


def send_images_from_array(image, canddate_TFL_lst):
    print("send cropped Img")
    return np.array(list(map(lambda x : [crop_notfl_img(image,x)] ,canddate_TFL_lst))), canddate_TFL_lst
    #     # res = get_lable_data(color_path, image_array)
    #     #if count <= 0:
    #     #    break
    # print('\n--Start Saving to binary--')
    # save_binary_file(current_labels)

def load_images_from_train():
    files_to_read = glob.glob(r'./gtFine_trainvaltest/gtFine/train/*/*polygons.json')
    count = 2
    labeled_list = []
    for file_path in files_to_read:
        count -= 1

        # image_array = np.array(Image.open(file_path), dtype='uint8')
        # after getting image array, we need to get the color path of this array
        # getting the directory name
        dic_with_file_path = '/'.join(file_path.split('\\')[-2:])
        file_change = dic_with_file_path.replace('gtFine_polygons.json', 'leftImg8bit.png')
        color_path = rf'./leftImg8bit_trainvaltest/leftImg8bit/train/{file_change}'
        print('Loading file:', color_path)
        image = np.array(Image.open(color_path), dtype='uint8')
        tfl_in_img = open_json(file_path, image)

        current_labels = list(map(lambda x: [x, 1], tfl_in_img))
        for i in range(len(tfl_in_img)):
            x = random.randint(0, image.shape[0])
            y = random.randint(0, image.shape[1])
            current_labels.append([crop_notfl_img(image, [x, y]), 0])
        labeled_list.extend(current_labels)
        # res = get_lable_data(color_path, image_array)
        #if count <= 0:
        #    break
    print('\n--Start Saving to binary--')
    save_binary_file(labeled_list)


if __name__ == '__main__':
    load_images_from_train()

    data = load_binary_data()
    print(data)
