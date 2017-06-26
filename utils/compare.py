import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tools
import scipy.misc
import numpy as np


def data2fig(which, samples, size=4):

    fig = plt.figure(figsize=(size, size))
    gs = gridspec.GridSpec(size, size)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        if which:
            plt.imshow(get_img(sample, 128, 64), cmap='Greys_r')
        else:
            plt.imshow(Image.open(sample), cmap='Greys_r')
        if i == size * size - 1:
            break
    return fig


def save_img(which, sample_dir, image_dir, file_name, size=4):
    tools.make_dir_if_noe_exist(sample_dir)

    _samples = os.listdir(image_dir)
    samples = [image_dir + sample for sample in _samples]
    fig = data2fig(which, samples, size=size)
    plt.savefig('{}/{}.png'.format(sample_dir, file_name), bbox_inches='tight')
    plt.close(fig)

    pass


def get_img(img_path, crop_h, resize_h):
    img = scipy.misc.imread(img_path).astype(np.float)
    # crop resize
    crop_w = crop_h
    # resize_h = 64
    resize_w = resize_h
    h, w = img.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    cropped_image = scipy.misc.imresize(img[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])

    return cropped_image


def save_mnist():
    save_img(which=False, sample_dir="../Samples/mnist-test", image_dir="../Datas/mnist-test/", file_name="compare-mnist", size=3)
    pass


def save_celebA():
    save_img(which=True, sample_dir="../Samples/celebA-test", image_dir="../Datas/celebA-test/", file_name="compare-celebA", size=5)
    pass

if __name__ == "__main__":
    save_celebA()
