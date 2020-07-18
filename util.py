import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from metrics import sCC
from metrics import ERGAS as ergas
from metrics import sam2 as sam


def save_figure(losses, path, epoch, label):
    # plt.plot(losses_d, label=label, color='b')
    #colors = ['r', 'b', 'm', 'y', 'g']
    # try:
        # if isinstance(losses[0], list):
            # for loss,c in zip(losses, colors):
                #plt.plot(loss, label=label, color=c)

    # except:
    if len(losses) == 2:
        plt.plot(losses[0], label='adv-loss', color='r')
        plt.plot(losses[1], label='recon-loss', color='g')

    else:
        plt.plot(losses, label=label, color='r')
        plt.title("Experiment: {} -- {}: {}".format(path, label, epoch))

    plt.legend()
    plt.savefig("results-{}/epoch{}-{}-loss.pdf".format(path, epoch, label,))
    plt.close()

def scale_range(input, min, max):
    input += -(np.min(input))
    input /= (1e-9 + np.max(input) / (max - min + 1e-9))
    input += min
    return input

def rgb2gray(rgb):
    r, g, b, nir = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], rgb[:, :, 3]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = 0.25 * r + 0.25 * g + 0.25 * b + 0.25 * nir
    return gray

def visualize_tensor(imgs, epoch, it, name):
    fname = "tensors-{}/{}/{}-{}.jpg".format(opt.savePath, epoch, it, name)
    vutils.save_image(
        tensor=imgs, filename=fname, normalize=True, nrow=imgs.size()[0] // 2)

def avg_metric(target, prediction, metric):
    sum = 0
    batch_size = len(target)
    for i in range(batch_size):
        sum += metric(np.transpose(target.data.cpu().numpy()
                                   [i], (1, 2, 0)), np.transpose(prediction.data.cpu().numpy()[i], (1, 2, 0)))
    return sum/batch_size

def show_image(im):
  if len(im.shape) == 2:
    im = scale_range(im, 0, 255).astype(np.uint8)
    plt.figure(figsize=(16, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(im,cmap='gray')
    plt.show()

  elif len(im.shape) == 3:
    im = np.array([scale_range(i, 0, 255) for i in im.transpose((2,0,1))]).transpose(1,2,0)[...,:3].astype(np.uint8)
    plt.figure(figsize=(16, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(im)
    plt.show()

def patch_sixteen(images):
    size = images.shape[-1]
    patched = np.zeros((images.shape[1],images.shape[2] * 4, images.shape[3] * 4))
    for i in range(4):
        for j in range(4):
            patched[..., i * size:(i + 1) * size, j * size:(j + 1) * size] += images[i*4+j]
    return np.array(patched)

def divide_sixteen(image):
    size = image.shape[-1]/4
    divided = []
    for i in range(4):
        for j in range(4):
            divided += image[...,int(i*size):int((i+1)*size),int(j*size):int((j+1)*size)],
    return np.array(divided)