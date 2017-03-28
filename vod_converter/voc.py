"""
Ingestor and egestor for VOC formats.

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html
"""

import os
import shutil

from converter import Ingestor, Egestor
import xml.etree.ElementTree as ET


class VOCIngestor(Ingestor):
    def validate(self, root):
        path = f"{root}/VOC2012"
        for subdir in ["ImageSets", "JPEGImages", "Annotations"]:
            if not os.path.isdir(f"{path}/{subdir}"):
                return False, f"Expected subdirectory {subdir} within {path}"
            if not os.path.isfile(f"{path}/ImageSets/Main/trainval.txt"):
                return False, f"Expected main image set ImageSets/Main/trainval.txt to exist within {path}"
        return True, None

    def ingest(self, path):
        image_names = self._get_image_ids(path)
        return [self._get_image_detection(path, image_name) for image_name in image_names]

    def _get_image_ids(self, root):
        path = f"{root}/VOC2012"
        with open(f"{path}/ImageSets/Main/trainval.txt") as f:
            fnames = []
            for line in f.read().strip().split('\n'):
                cols = line.split()
                if len(cols) > 1:
                    score = cols[1]
                    if score != '1':
                        continue
                fnames.append(cols[0])
            return fnames

    def _get_image_detection(self, root, image_id):
        path = f"{root}/VOC2012"
        image_path = f"{path}/JPEGImages/{image_id}.jpg"
        if not os.path.isfile(image_path):
            raise Exception(f"Expected {image_path} to exist.")
        annotation_path = f"{path}/Annotations/{image_id}.xml"
        if not os.path.isfile(annotation_path):
            raise Exception(f"Expected annotation file {annotation_path} to exist.")
        tree = ET.parse(annotation_path)
        xml_root = tree.getroot()
        size = xml_root.find('size')
        segmented = xml_root.find('segmented').text == '1'
        segmented_path = None
        if segmented:
            segmented_path = f"{path}/SegmentationObject/{image_id}.png"
            if not os.path.isfile(segmented_path):
                raise Exception(f"Expected segmentation file {segmented_path} to exist.")
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)
        return {
            'image': {
                'id': image_id,
                'path': image_path,
                'segmented_path': segmented_path,
                'width': image_width,
                'height': image_height
            },
            'detections': [self._get_detection(node) for node in xml_root.findall('object')]
        }

    def _get_detection(self, node):
        bndbox = node.find('bndbox')
        return {
            'label': node.find('name').text,
            'top': float(bndbox.find('ymin').text) - 1,
            'left': float(bndbox.find('xmin').text) - 1,
            'right': float(bndbox.find('xmax').text) - 1,
            'bottom': float(bndbox.find('ymax').text) - 1,
        }


class VOCEgestor(Egestor):

    def expected_labels(self):
        return {
            'aeroplane': [],
            'bicycle': [],
            'bird': [],
            'boat': [],
            'bottle': [],
            'bus': [],
            'car': [],
            'cat': [],
            'chair': [],
            'cow': [],
            'diningtable': [],
            'dog': [],
            'horse': [],
            'motorbike': [],
            'person': ['pedestrian'],
            'pottedplant': [],
            'sheep': [],
            'sofa': [],
            'train': [],
            'tvmonitor': []
        }

    def egest(self, *, image_detections, root):
        image_sets_path = f"{root}/VOC2012/ImageSets/Main"
        images_path = f"{root}/VOC2012/JPEGImages"
        annotations_path = f"{root}/VOC2012/Annotations"
        segmentations_path = f"{root}/VOC2012/SegmentationObject"
        segmentations_dir_created = False

        for to_create in [image_sets_path, images_path, annotations_path]:
            os.makedirs(to_create, exist_ok=True)

        for image_detection in image_detections:
            image = image_detection['image']
            image_id = image['id']
            src_extension = image['path'].split('.')[-1]
            shutil.copyfile(image['path'], f"{images_path}/{image_id}.{src_extension}")

            with open(f"{image_sets_path}/trainval.txt", 'a') as out_image_index_file:
                out_image_index_file.write(f'{image_id}\n')

            if image['segmented_path'] is not None:
                if not segmentations_dir_created:
                    os.makedirs(segmentations_path)
                    segmentations_dir_created = True
                shutil.copyfile(image['segmented_path'], f"{segmentations_path}/{image_id}.png")

            xml_root = ET.Element('annotation')
            add_text_node(xml_root, 'filename', f"{image_id}.{src_extension}")
            add_text_node(xml_root, 'folder', 'VOC2012')
            add_text_node(xml_root, 'segmented', int(segmentations_dir_created))

            add_sub_node(xml_root, 'size', {
                'depth': 3,
                'width': image['width'],
                'height': image['height']
            })
            add_sub_node(xml_root, 'source', {
                'annotation': 'Dummy',
                'database': 'Dummy',
                'image': 'Dummy'
            })

            for detection in image_detection['detections']:
                x_object = add_sub_node(xml_root, 'object', {
                    'name': detection['label'],
                    'difficult': 0,
                    'occluded': 0,
                    'truncated': 0,
                    'pose': 'Unspecified'
                })
                add_sub_node(x_object, 'bndbox', {
                    'xmin': detection['left'] + 1,
                    'xmax': detection['right'] + 1,
                    'ymin': detection['top'] + 1,
                    'ymax': detection['bottom'] + 1
                })

            ET.ElementTree(xml_root).write(f"{annotations_path}/{image_id}.xml")


def add_sub_node(node, name, kvs):
    subnode = ET.SubElement(node, name)
    for k, v in kvs.items():
        add_text_node(subnode, k, v)
    return subnode


def add_text_node(node, name, text):
    subnode = ET.SubElement(node, name)
    subnode.text = f"{text}"
    return subnode







