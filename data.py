import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps, ImageEnhance
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class RandomTranslate(object):
    def __call__(self, img):
        mag = random.uniform(-0.25, 0.25)
        mat = (
            1, 0, mag * img.size[0],
            0, 1, 0
        ) if random.choice([0, 1]) else (
            1, 0, 0,
            0, 1, mag * img.size[1]
        )
        return img.transform(
            img.size, Image.AFFINE, mat, fillcolor=(128, 128, 128)
        )


class RandSharpness(object):
    def __call__(self, x: Image.Image):
        mag = random.uniform(-0.8, 0.8)
        return ImageEnhance.Sharpness(x).enhance(
            1 + mag
        )


class AutoContrast(object):
    def __call__(self, img):
        return ImageOps.autocontrast(img)


class Scene15Set(ImageFolder):
    def __init__(
            self, root_dir_path, train, vgg=False,
            rot=10, scale_ratio=0.6, val_crop=True
    ):
        root_dir_path = os.path.join(
            root_dir_path, 'train' if train else 'test'
        )
        
        r1 = scale_ratio if scale_ratio < 1 else 1 / scale_ratio
        r2 = 1 / scale_ratio
        taget_im_size = 224 if vgg else 256
        if train:
            aug = [
                transforms.RandomResizedCrop(taget_im_size, scale=(0.32, 1), ratio=(r1, r2)),
                transforms.RandomChoice((RandSharpness(), AutoContrast())),
                RandomTranslate(),
                transforms.ColorJitter(0.32, 0.32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(rot),
            ]
        else:
            if val_crop:
                aug = [transforms.Resize(round(taget_im_size * 1.143)), transforms.CenterCrop(taget_im_size)]
            else:
                aug = [transforms.Resize(taget_im_size)]
        
        if vgg:
            normalize = [
                transforms.ToTensor(),
                lambda im: im.flip(0).mul_(255).sub_(torch.as_tensor([103.939, 116.779, 123.68])[:, None, None]),
            ]
        else:
            normalize = [
                transforms.ToTensor(),
                lambda im: im.sub_(0.45567986).mul_(1 / 0.25347006),
            ]
        
        super(Scene15Set, self).__init__(
            root_dir_path,
            transform=transforms.Compose(aug + normalize)
        )


def calc_stats():
    t = ImageFolder(r'C:\Users\16333\Desktop\PyCharm\cv_course\proj_3\homework_set_3\data\train')
    data = np.concatenate([np.asarray(x[0]).reshape(-1, 3) for x in t])
    m, s = data.mean(axis=0) / 255., data.std(axis=0) / 255.
    print(f'mean={m}, std={s}')
    # mean=[0.45567986 0.45567986 0.45567986], std=[0.25347006 0.25347006 0.25347006]
    
    wh = defaultdict(int)
    avg_w, avg_h, avg_ratio = 0, 0, 0
    max_reso = (0, 0)
    for im, _ in t:
        w, h = im.size
        wh[(w, h)] += 1
        avg_w += w
        avg_h += h
        avg_ratio += w / h if w < h else h / w
        if w * h > max_reso[0] * max_reso[1]:
            max_reso = (w, h)
    avg_w /= len(t)
    avg_h /= len(t)
    avg_ratio /= len(t)
    print(f'avg_w={avg_w:.1f}, avg_h={avg_h:.1f}\n'
          f'max_reso={max_reso}\n'
          f'avg_ratio={avg_ratio:.2f}, ratio= {min(i[0] / i[1] for i in wh.keys()):.2f} ~ {max(i[0] / i[1] for i in wh.keys()):.2f}\n'
          f'wh={sorted(wh.items(), key=lambda i: i[1], reverse=True)[:5]}')
    # avg_w=275.9, avg_h=242.8
    # max_reso=(220, 411)
    # ratio= 0.5352798053527981 ~ 1.8681818181818182
    # wh=[((256, 256), 800), ((293, 220), 151), ((330, 220), 125), ((220, 293), 24), ((294, 220), 14)]


def visualize():
    import matplotlib.pyplot as plt
    
    t = Scene15Set(r'C:\Users\16333\Desktop\PyCharm\cv_course\proj_3\homework_set_3\data', train=True, vgg=False)
    for idx in range(1320, len(t), 2):
        im1 = torch.stack([t[idx][0] for _ in range(32)])
        # im2 = torch.stack([t[idx+1][0] for _ in range(16)])
        # ims = torch.cat((im1, im2))
        ims = im1
        grids: torch.Tensor = torchvision.utils.make_grid(ims, padding=10).mul_(0.25347006).add(0.45567986)
        plt.imshow(np.transpose(grids.numpy(), (1, 2, 0)), interpolation='nearest')
        plt.show()


if __name__ == '__main__':
    # calc_stats()
    visualize()
