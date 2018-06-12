import random
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

from utils.encoder import DataEncoder

roaddamage_label_names = (
    'D00', 'D01', 'D10', 'D11',
    'D20', 'D40', 'D43', 'D44'
)

class road_datareader(Dataset):

    def __init__(self, data_dir, split='train'):

        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                "split must be either of 'train', 'traival', or 'val'"
            )

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        if split == 'train':
            self.train = True
        else:
            self.train = False

        self.data_encoder = DataEncoder()

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.img_size = 300
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])

        print('Load dataset with {} examples'.format(len(self.ids)))

        self.bbox = []
        self.labels = []

        for i in range(len(self.ids)):
            id_ = self.ids[i]

            anno = ET.parse(
                os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))

            bbox = []
            label = []

            for obj in anno.findall('object'):
                bndbox_anno = obj.find('bndbox')

                name = obj.find('name').text.strip()
                if name not in roaddamage_label_names:
                    continue

                label.append(roaddamage_label_names.index(name))

                bbox.append([
                    int(bndbox_anno.find(tag).text) - 1
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

            self.bbox.append(torch.Tensor(bbox))
            self.labels.append(torch.LongTensor(label))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        boxes = self.bbox[idx].clone()
        labels = self.labels[idx]
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)

        img_file = os.path.join(self.data_dir, 'JPEGImages', self.ids[idx] + '.jpg')

        img = Image.open(img_file)

        # Data augmentation while training.
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # Scale bbox locaitons to [0,1].
        w, h = img.size
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        img = img.resize((self.img_size, self.img_size))

        img = self.transform(img)
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.
        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).
        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''Randomly crop the image and adjust the bbox locations.
        For more details, see 'Chapter2.2: Data augmentation' of the paper.
        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) bbox labels, sized [#obj,].
        Returns:
          img: (PIL.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.size
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                w = random.randrange(int(0.1 * imw), imw)
                h = random.randrange(int(0.1 * imh), imh)

                if h > 2 * w or w > 2 * h:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x + w, y + h]])

                center = (boxes[:, :2] + boxes[:, 2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
                mask = (center > roi2[:, :2]) & (center < roi2[:, 2:])  # [N,2]
                mask = mask[:, 0] & mask[:, 1]  # [N,]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

                iou = self.data_encoder.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue

                img = img.crop((x, y, x + w, y + h))
                selected_boxes[:, 0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:, 2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 3].add_(-y).clamp_(min=0, max=h)

                return img, selected_boxes, labels[mask]