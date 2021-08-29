try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from scipy import ndimage
    from PIL import Image
    import scipy.ndimage
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise



def plot(data, title):
    plot.i += 1
    if plot.ax is None:
        plot.ax = plt.subplot(2, 2, plot.i)
    else:
        plt.subplot(2, 2, plot.i, sharex=plot.ax, sharey=plot.ax)
    plt.imshow(data)
    plt.gray()
    plt.title(title)


plot.i = 0
plot.ax = None


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    data = c_image[:, :, 0].astype(float)
    data1 = c_image[:, :, 1].astype(float)
    data2 = c_image[:, :, 2].astype(float)

    kernel_3 = np.array([[81., 68., 62., 90., 106., 119., 121., 128., 132., 135., 131.,
                          129., 121., 115., 96., 84., 71., 77.],
                         [92., 77., 88., 99., 114., 127., 205., 217., 255., 247., 242.,
                          207., 180., 140., 124., 95., 92., 100.],
                         [98., 104., 113., 137., 138., 188., 228., 255., 255., 255., 255.,
                          255., 219., 184., 139., 130., 108., 111.],
                         [120., 128., 200., 209., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 251., 236., 186., 160., 133.],
                         [129., 141., 214., 223., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 210., 186., 143.],
                         [149., 162., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 255., 220., 159.],
                         [157., 176., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 255., 244., 173.],
                         [170., 198., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 255., 255., 177.],
                         [172., 218., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 255., 255., 184.],
                         [173., 211., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 255., 255., 178.],
                         [165., 212., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 255., 250., 168.],
                         [147., 194., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 255., 229., 157.],
                         [142., 175., 255., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 249., 205., 143.],
                         [129., 153., 203., 255., 255., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 255., 222., 179., 129.],
                         [110., 120., 114., 203., 250., 255., 255., 255., 255., 255., 255.,
                          255., 255., 255., 252., 176., 147., 115.],
                         [99., 107., 118., 106., 160., 233., 255., 255., 255., 255., 255.,
                          255., 255., 243., 200., 145., 124., 98.],
                         [82., 86., 101., 113., 100., 163., 216., 255., 255., 255., 255.,
                          222., 183., 169., 127., 105., 80., 79.],
                         [71., 69., 76., 84., 93., 88., 124., 200., 186., 213., 185.,
                          160., 140., 124., 104., 69., 67., 58.]])

    #kernel = (kernel_3 - kernel_3.mean()) / 1e5

    kernel_2 = (kernel_3 - kernel_3.mean()) / 1e5

    kernel_3 = (kernel_3 - kernel_3.mean()) / 1e5

    highpass_3x3 = ndimage.convolve(data, kernel_3, mode='reflect')
    highpass_3x31 = ndimage.convolve(data1, kernel_2, mode='reflect')
    highpass_3x32 = ndimage.convolve(data2, kernel_3, mode='reflect')

    filtered = maximum_filter(highpass_3x3, (5, 5))

    y = filtered.shape[0]
    x = filtered.shape[1]
    red_x = []
    red_y = []
    green_x = []
    green_y = []
    maximum_number = 0
    maximum_number1 = 0
    maximum_number2 = 0
    for i in range(x):
        for j in range(y):
            if highpass_3x3[j, i] > maximum_number:
                maximum_number = highpass_3x3[j, i]
            if highpass_3x31[j, i] > maximum_number1:
                maximum_number1 = highpass_3x31[j, i]
            if highpass_3x32[j, i] > maximum_number2:
                maximum_number2 = highpass_3x32[j, i]

    i = 0
    while i < x:
        j = 0
        while j < y:
            if highpass_3x3[j, i] >= maximum_number - 1.5 or highpass_3x31[j, i] >= maximum_number1 - 1.5 or \
                    highpass_3x32[j, i] >= maximum_number2 - 1.5:
                red_x.append(i)
                red_y.append(j)
                i += 40
                j += 40
            j += 1
        i += 1
    return red_x, red_y, green_x, green_y


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'rx', markersize=4)
    plt.plot(green_x, green_y, 'g+', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "./aachen"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
