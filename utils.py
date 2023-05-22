import torch
import numpy as np
from os import listdir
import cv2
from scipy import ndimage
from torchvision import transforms

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append( file)
        elif name.endswith('.jpg'):
            images.append(file)
        elif name.endswith('.jpeg'):
            images.append(file)
        name1 = name.split('.')
        names.append(name1[0])
    return images

def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = ndimage.imread(path, mode='L')
    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode) 
        w, h = image.shape[0], image.shape[1]
        w_s = 256 - w % 256
        h_s = 256 - h % 256
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,
                                     value=128)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    cv2.imwrite(path, ori)
