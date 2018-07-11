#!/usr/bin/python
#coding: utf-8
import sys
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def get_files_in_dir(dir_path, exts):
    files = []
    for ext in exts:
        sub_files = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,f)) and f.endswith(ext)]
        files += sub_files
    return files

def parse_anno(anno_file):
    tree = ET.parse(anno_file)
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = ['' for _ in range(num_objs)]

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = obj.find('name').text.lower().strip()

    return boxes, gt_classes

def render_anno(img_file, anno_file):
    boxes, gt_classes = parse_anno(anno_file)
    img = cv2.imread(img_file, 1)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_name = gt_classes[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    return img

def main(argv):
    if len(argv) != 2:
        print('usage:\n\t%s [voc_root_dir]' % argv[0])
        sys.exit(-1)
    voc_root_dir = argv[1]
    img_dir = os.path.join(voc_root_dir, 'JPEGImages')
    anno_dir = os.path.join(voc_root_dir, 'Annotations')
    img_files = get_files_in_dir(img_dir, ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'])
    anno_files = get_files_in_dir(anno_dir, ['.xml'])
    anno_files = dict([(anno_file, 1) for anno_file in anno_files])
    for img_file in img_files:
        anno_file = img_file.replace('JPEGImages', 'Annotations')
        anno_file = '.'.join(anno_file.split('.')[:-1]) + '.xml'
        if anno_files.get(anno_file) is None:
            print('anno file is not exists.')
            continue
        img = render_anno(img_file, anno_file)
        if img is None:
            print('image is invalidate.')
            continue
        print('image: ' + img_file)
        print('anno: ' + anno_file)
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 27:
            break

if __name__ == '__main__':
    main(sys.argv)
