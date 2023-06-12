import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import torchvision
from torchvision import transforms

import numpy as np
import cv2
import os

from common import prepare_vgg_model

device = torch.device('cpu')


class NumpyDataset(Dataset):
    def __init__(self, np_array):
        self.data = np_array.astype(np.uint8)
        self.dataset_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, index: int):
        img = self.data[index]
        img = self.dataset_transforms(img)
        return img

    def __len__(self):
        return len(self.data)


def predict(model, data_list, bboxes):
    input_dataset = NumpyDataset(np.array(data_list))

    dataloader = DataLoader(input_dataset, 
        batch_size=128, 
        shuffle=False)

    bboxes[:, 2:4] = bboxes[:, 2:4] + bboxes[:, 0:2] # Convert to x1, y1, x2, y2
    bboxes = torch.from_numpy(bboxes)

    pred_vals, pred_inds = None, None

    with torch.set_grad_enabled(False):
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            val, ind = torch.max(outputs, dim=1)
            pred_vals = val if pred_vals is None else torch.cat((pred_vals, val), dim=0)
            pred_inds = ind if pred_inds is None else torch.cat((pred_inds, ind), dim=0)

        # Filter "not a number" class
        kept_inds = (pred_inds != 10).nonzero().squeeze()
        pred_vals, pred_inds, bboxes = pred_vals[kept_inds], pred_inds[kept_inds], bboxes[kept_inds]

        # Non-maxima suppression
        kept_inds = torchvision.ops.nms(bboxes.float(), pred_vals, iou_threshold=0.01)
        pred_vals, pred_inds, bboxes = pred_vals[kept_inds], pred_inds[kept_inds], bboxes[kept_inds]

        # Filter pred_vals
        kept_inds = (pred_vals > 5.0).nonzero().squeeze()
        pred_vals, pred_inds, bboxes = pred_vals[kept_inds], pred_inds[kept_inds], bboxes[kept_inds]

        pred_inds = pred_inds.detach().numpy()
        bboxes = bboxes.detach().numpy()

    return bboxes, pred_inds


def get_mser_bounding_boxes(img):
    # Used to filter bounding boxes that have width greater than WIDTH_HEIGHT_RATIO*height
    WIDTH_HEIGHT_RATIO = 1

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser_detector = cv2.MSER_create(_min_area=100, _max_area=2000, _delta=10)
    _, bboxes1 = mser_detector.detectRegions(img_gray)
    _, bboxes2 = mser_detector.detectRegions(img)

    bboxes = np.concatenate((bboxes1, bboxes2), axis=0)

    bboxes = np.unique(bboxes, axis=0)
    bboxes = bboxes[(bboxes[:, 2] / bboxes[:, 3] < WIDTH_HEIGHT_RATIO)]

    return bboxes


def get_cutout_list(img, bboxes):
    cutout_list = []
    img_y, img_x, _ = img.shape
    for x, y, width, height in bboxes:
        size = max(width, height)

        if height > width:
            diff = (height - width) // 2
        else:
            diff = (width - height) // 2
        
        # Use squared crop
        x = max(0, x - diff)
        xend = min(x + size, img_x)
        yend = min(y + size, img_y)
        xdiff, ydiff = xend - x, yend - y

        crop = np.zeros((size, size, 3), dtype=np.uint8)
        crop[0:ydiff, 0:xdiff, :] = img[y:yend, x:xend, :]
        crop = cv2.resize(crop, (32, 32))

        cutout_list.append(crop)
    return cutout_list


def grade_images(model, graded_img_list, in_folder, out_folder):
    for img_ind, img_filename in enumerate(graded_img_list):
        img = cv2.imread(os.path.join(in_folder, img_filename))

        # Resize image
        img_y, img_x, _ = img.shape
        ratio = 250 / img_y
        newsize_y, newsize_x = int(img_y * ratio), int(img_x * ratio)
        img = cv2.resize(img, dsize=(newsize_x, newsize_y))
        
        bboxes = get_mser_bounding_boxes(img)
        # bboxes[:, 2:4] = bboxes[:, 2:4] + bboxes[:, 0:2] # Convert to x1, y1, x2, y2

        cutout_list = get_cutout_list(img, bboxes)
        bboxes, pred = predict(model, cutout_list, bboxes)

        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            label = pred[i]

            if label != 10:  # 10 is "not a number" class
                cv2.rectangle(img, (x1, y1), (x2, y2), 
                            color=(0, 255, 0), thickness=1)
                
                cv2.putText(img, str(label), (x1-10, y1-5), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5,
                            color=(135, 135, 255), thickness=2)

        cv2.imwrite(os.path.join(out_folder, f'{img_ind + 1}.jpg'), img)


if __name__ == '__main__':
    model = prepare_vgg_model(pretrained=False)
    
    model_state_dict = torch.load('jtan319_model.pt', 
                                  map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)

    graded_img_list = ['test4.jpg', 'test5.jpg', 'test8.jpg', 'test9.jpg', 'test11.jpg']

    in_folder = 'test_images'
    out_folder = 'graded_images'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    grade_images(model, graded_img_list, in_folder, out_folder)
