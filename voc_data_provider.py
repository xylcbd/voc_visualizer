#coding: utf-8

import os
import xml.etree.ElementTree as ET
import cv2
import copy
import sys

########################### main functions ############################
class PascalVOCDataProvider(object):
    SET_TRAIN = 'train'
    SET_VAL = 'val'
    SET_TEST = 'test'

    ALL_CLASSES = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    )

    @staticmethod
    def label2name(label):
        name = PascalVOCDataProvider.ALL_CLASSES[label]
        return name

    @staticmethod
    def name2label(name):
        label = PascalVOCDataProvider.ALL_CLASSES.index(name)
        return label

    def __init__(self, dataset_root_dir = '~/data/VOCdevkit/VOC2007/', set_name=SET_TRAIN):
        self.dataset_root_dir = dataset_root_dir
        self.image_dir = os.path.join(self.dataset_root_dir, 'JPEGImages')
        self.anno_dir = os.path.join(self.dataset_root_dir, 'Annotations')
        self.IDs_file = os.path.join(self.dataset_root_dir, 'ImageSets', 'Main', set_name + '.txt')
        self.list_IDs = [line.strip() for line in open(self.IDs_file).readlines()]
        assert len(self.list_IDs) > 0

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        assert index >= 0 or index < len(self.list_IDs)
        ID = self.list_IDs[index]
        return self.load(ID)

    def load(self, ID):
        image_file = os.path.join(self.image_dir, ID+'.jpg')
        anno_file = os.path.join(self.anno_dir, ID+'.xml')

        #read image
        image = self.load_image(image_file)

        #parse xml
        bboxes, labels = self.load_anno(anno_file)

        return image, bboxes, labels

    def load_image(self, image_file):
        image = cv2.imread(image_file, 1)
        return image

    def load_anno(self, anno_file):
        anno = ET.parse(anno_file)
        bboxes = list()
        labels = list()
        for obj in anno.findall('object'):
            if int(obj.find('difficult').text) == 1:
                continue
            bndbox_anno = obj.find('bndbox')
            bboxes.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            labels.append(PascalVOCDataProvider.name2label(name))
        return bboxes, labels

########################### debug functions ############################
def render_anno(image, bboxes, labels):
    sh_image = copy.deepcopy(image)
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        cls_name = labels[i]
        cv2.rectangle(sh_image, (x1, y1), (x2, y2), (255,0,0), 2)
    return sh_image

def print_details(data_provider, set_name):
    print('--------------')
    print('length of ' + set_name + ' set: ', len(data_provider))
    _, (image, bboxes, labels) = enumerate(data_provider).next()
    print('image shape: ', image.shape)
    print('bboxes: ', bboxes)
    print('labels: ', labels)
    print('names: ', [PascalVOCDataProvider.label2name(label) for label in labels])
    print('--------------\n')
    sh_image = render_anno(image, bboxes, labels)
    cv2.imshow('image', sh_image)
    cv2.waitKey(0)

if __name__=='__main__':
    if len(sys.argv) != 2:
        print('usage:\n\t%s [voc_root_dir]' % sys.argv[0])
        sys.exit(-1)
    voc_root_dir = sys.argv[1]

    train_data_provider = PascalVOCDataProvider(dataset_root_dir=voc_root_dir, set_name=PascalVOCDataProvider.SET_TRAIN)
    print_details(train_data_provider, 'train')

    val_data_provider = PascalVOCDataProvider(dataset_root_dir=voc_root_dir, set_name=PascalVOCDataProvider.SET_VAL)
    print_details(val_data_provider, 'val')

    test_data_provider = PascalVOCDataProvider(dataset_root_dir=voc_root_dir, set_name=PascalVOCDataProvider.SET_TEST)
    print_details(test_data_provider, 'test')
