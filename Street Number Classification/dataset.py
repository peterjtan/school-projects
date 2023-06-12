from torchvision import transforms
from torchvision.datasets import SVHN
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import h5py  # for .mat files with MATLAB 7.3 format
import numpy as np
import cv2
import pickle
import os
from tqdm.autonotebook import trange



class MySVHNDatasetBase(object):
    def __init__(self, root, mode, download=True):
        self.configs = {
            'train': ('http://ufldl.stanford.edu/housenumbers/train.tar.gz', 'train.tar.gz'),
            'test': ('http://ufldl.stanford.edu/housenumbers/test.tar.gz', 'test.tar.gz')
        }

        self.root = root
        self.mode = mode

        if download:
            self.download()
        else:
            filename = self.configs[self.mode][1]
            if not check_integrity(self.root, filename):
                raise RuntimeError('File corrupted. You can use download=True to redownload.')
    
    def download(self):
        url, filename = self.configs[self.mode] 
        if check_integrity(os.path.join(self.root, filename)):
            print('File is already downloaded. ')
        else:
            download_and_extract_archive(url, self.root)

    def init_digit_struct(self):
        saved_pkl_path = os.path.join(self.root, self.mode, 'digitStruct.pkl')
        if os.path.isfile(saved_pkl_path):
            print('\'digitStruct.pkl\' found. ')
            with open(saved_pkl_path, 'rb') as f:
                pkl_obj = pickle.load(f)
                filesnames, bboxes = pkl_obj['filenames'], pkl_obj['bboxes']
            return filesnames, bboxes
        else:
            print('\'digitStruct.pkl\' not found. Construct arrays from .mat file. ')
            filenames = []
            bboxes = []
            with h5py.File(os.path.join(self.root, self.mode, 'digitStruct.mat'), 'r') as f:
                size = f['/digitStruct/name'].size
                for i in trange(size):
                    filenames.append(self.get_name(f, i))
                    bboxes.append(self.get_bbox(f, i))
            
            pkl_obj = {
                'filenames': filenames,
                'bboxes': bboxes
            }

            with open(os.path.join(self.root, self.mode, 'digitStruct.pkl'), 'wb') as f:
                pickle.dump(pkl_obj, f)

            return filenames, bboxes
    
    def get_name(self, h5f, index):
        name_ref = h5f['/digitStruct/name'][index][0]
        obj = h5f[name_ref][:]
        str = obj.squeeze().astype(np.uint8).tostring().decode('ascii')
        return str

    def get_bbox(self, h5f, index):
        bbox_ref = h5f['/digitStruct/bbox'][index][0]
        bbox = h5f[bbox_ref]

        dict = {key: [] for key in ['top', 'left', 'width', 'height', 'label']}
        for key in bbox.keys():
            bbox_elem = bbox[key]
            if bbox_elem.shape[0] == 1:
                dict[key].append(int(bbox_elem[0, 0]))
            else:
                for i in range(bbox_elem.shape[0]):
                    dict[key].append(int(h5f[bbox_elem[i, 0]][:].item()))
        
        res = []
        for i in range(len(dict['top'])):
            res.append({'top': dict['top'][i], 
                        'left': dict['left'][i], 
                        'width': dict['width'][i],
                        'height': dict['height'][i],
                        'label': dict['label'][i]})
        return res



class MySVHNDataset(VisionDataset, MySVHNDatasetBase):
    def __init__(self, root, mode, transform=None, target_transform=None, download=True):
        print(f'Initializing MySVHNDataset with mode: {mode}...')
        MySVHNDatasetBase.__init__(self, root, mode, download)
        VisionDataset.__init__(self, root, transform=transform, target_transform=target_transform)

        self.info_map_list = []
        filenames, bboxes = self.init_digit_struct()

        for i in trange(len(filenames)):
            filename = filenames[i]
            bbox_list = bboxes[i]
            for bbox in bbox_list:
                self.info_map_list.append((filename, bbox))

    def __getitem__(self, index):
        filename = self.info_map_list[index][0]
        infomap = self.info_map_list[index][1]
        x, y = infomap['left'], infomap['top']
        width, height = infomap['width'], infomap['height']
        label = infomap['label']
        sz = max(width, height)

        # Change the image to be squared with dark background
        res = np.zeros((sz, sz, 3), dtype=np.uint8)
        img = cv2.imread(os.path.join(self.root, self.mode, filename))
        img_y, img_x, _ = img.shape

        ybegin, yend = max(0, y), min(y + height, img_y)
        xbegin, xend = max(0, x), min(x + width, img_x)
        ydiff, xdiff = yend - ybegin, xend - xbegin
        # print(f'ybegin = {ybegin}, yend = {yend}, xbegin = {xbegin}, xend = {xend}, ydiff = {ydiff}, xdiff = {xdiff}')
        res[0:ydiff, 0:xdiff, :] = img[ybegin:yend, xbegin:xend, :]

        # Resize
        res = cv2.resize(res, (32, 32))

        res = res.astype(np.uint8)
        
        if self.transform is not None:
            res = self.transform(res)
        
        return res, label

    def __len__(self):
        return len(self.info_map_list)


class MySVHNNotNumberDataset(VisionDataset, MySVHNDatasetBase):
    def __init__(self, root, mode, transform=None, target_transform=None, download=True):
        print(f'Initializing MySVHNNotNumberDataset with mode: {mode}...')
        MySVHNDatasetBase.__init__(self, root, mode, download)
        VisionDataset.__init__(self, root, transform=transform, target_transform=target_transform)

        filenames, bboxes = self.init_digit_struct()
        self.index_list = []
        self.image_list = []

        self.window_size = 150
        self.skip_pixels = 15

        for i in trange(len(filenames)):
            filename = filenames[i]
            bbox_list = bboxes[i]

            img = cv2.imread(os.path.join(self.root, self.mode, filename))
            img_y, img_x, _ = img.shape

            for bbox in bbox_list:
                x, y, width, height = bbox['left'], bbox['top'], bbox['width'], bbox['height']

                x, y = max(0, x), max(0, y)
                xend, yend = min(img_x, x + width), min(img_y, y + height)

                # Black out street numbers
                img[y:yend, x:xend, :] = 0

            self.image_list.append(img)

            ind_y, ind_x = max(0, img_y - self.window_size), max(0, img_x - self.window_size)
            for y in range(0, ind_y, self.skip_pixels):
                for x in range(0, ind_x, self.skip_pixels):
                    self.index_list.append((i, y, x))

    def __getitem__(self, index):
        index_tup = self.index_list[index]
        ind_img, ind_y, ind_x = index_tup

        img = self.image_list[ind_img]
        img_y, img_x, _ = img.shape

        if (ind_y >= img_y or ind_x >= img_x):
            cutout = np.zeros((32, 32, 3))
        else:
            y, x = max(0, ind_y), max(0, ind_x)
            yend, xend = min(ind_y + self.window_size, img_y), min(ind_x + self.window_size, img_x)

            if yend - y <= 0 or xend - x <= 0:
                print(f'ind_y = {ind_y}, ind_x = {ind_x}, img_y = {img_y}, img_x = {img_x}')

            cutout = img[y:yend, x:xend, :]
            cutout = cv2.resize(cutout, (32, 32))
        
        if self.transform is not None:
            cutout = self.transform(cutout)
        
        return cutout

    def __len__(self):
        return len(self.index_list)


class GaussianNoise(object):
    def __init__(self, noise_std=1., noise_mean=0.): 
        self.noise_std = noise_std
        self.noise_mean = noise_mean
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.noise_std + self.noise_mean


class MyDataset(Dataset):
    def __init__(self, is_train):
        dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomGrayscale(),
            transforms.RandomPerspective(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.20, saturation=0.20),
            # Gaussian noise
            transforms.RandomApply([GaussianNoise(noise_std=0.2, noise_mean=0.)], p=0.50),
            # Normalize based on ImageNet mean/std params
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        mode = 'train' if is_train else 'test'

        self.NOT_A_NUMBER_CLASS = 10

        self.svhn_dataset1 = MySVHNDataset('SVHN_dataset_fullsize', mode, 
                                           transform=dataset_transforms, download=True)

        self.svhn_dataset2 = SVHN('SVHN_Dataset_32by32', mode, 
                                  transform=dataset_transforms, download=True)

        self.not_number_dataset = MySVHNNotNumberDataset('SVHN_dataset_fullsize', mode, 
                                                         transform=dataset_transforms, download=True)

        self.len_svhn_dataset1 = len(self.svhn_dataset1)
        self.len_svhn_dataset2 = len(self.svhn_dataset2)
        self.len_not_number_dataset = len(self.not_number_dataset)

    def __len__(self):
        return self.len_svhn_dataset1 + self.len_svhn_dataset2 + self.len_not_number_dataset

    def __getitem__(self, index):
        if index < self.len_svhn_dataset1:
            return self.svhn_dataset1.__getitem__(index)
        elif index < (self.len_svhn_dataset1 + self.len_svhn_dataset2):
            return self.svhn_dataset2.__getitem__(index - self.len_svhn_dataset1)
        else:
            img = self.not_number_dataset.__getitem__(index - self.len_svhn_dataset1 - self.len_svhn_dataset2)
            target = self.NOT_A_NUMBER_CLASS
            return img, target


def prepare_dataset():
    """
    Prepare training and testing datasets. 
    Returns:
    * dataloader: A dictionary of keys 'train' and 'test' and values of 2
                  instances of torch.utils.data.DataLoader containing the
                  training and testing datasets. 
    * dataset_size: A dictionary of keys 'train' and 'test' containing the
                    lengths of the above datasets. 
    """
    train_dataset = MyDataset(is_train=True)
    test_dataset = MyDataset(is_train=False)

    dataloader_param = {
        'batch_size': 1024,
        'shuffle': True,
        'num_workers': 2,
    }
    dataloader = {
        'train': DataLoader(train_dataset, **dataloader_param),
        'test': DataLoader(test_dataset, **dataloader_param)
    }

    dataset_size = {
        'train': len(train_dataset),
        'test': len(test_dataset)
    }

    return dataloader, dataset_size
