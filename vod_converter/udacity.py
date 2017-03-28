"""
https://github.com/udacity/self-driving-car/tree/master/annotations


"""

import csv
import glob
import os
from PIL import Image

from collections import defaultdict


from converter import Ingestor


class UdacityCrowdAIIngestor(Ingestor):

    def validate(self, root):
        labels_path = f"{root}/labels.csv"
        if not os.path.isfile(labels_path):
            return False, f"Expected to find {labels_path}"
        return True, None

    def ingest(self, root):
        labels_path = f"{root}/labels.csv"
        image_labels = defaultdict(list)

        with open(labels_path) as labels_file:
            labels_csv = csv.reader(labels_file)
            next(labels_csv, None)  # skip header
            for idx, row in enumerate(labels_csv):
                image_labels[row[4]].append(row)

        image_detections = []
        for idx, image_path in enumerate(glob.glob(f"{root}/*.jpg")):
            f_name = image_path.split("/")[-1]
            f_image_labels = image_labels[f_name]
            fname_id = f_name.split('.')[0]

            image_width, image_height = _image_dimensions(image_path)

            def clamp_bbox(det):
                if det['right'] > image_width - 1:
                    det['right'] = image_width - 1
                if det['bottom'] > image_height - 1:
                    det['bottom'] = image_height - 1
                return det

            def valid_bbox(det):
                return det['right'] > det['left'] and det['bottom'] > det['top']

            detections = []

            for image_label in f_image_labels:
                x1, y1, x2, y2 = map(float, image_label[0:4])
                label = image_label[5]
                detections.append({
                    'label': label,
                    'left': x1,
                    'right': x2,
                    'top': y1,
                    'bottom': y2
                })

            filtered_detections = [clamp_bbox(det) for det in detections if valid_bbox(det)]
            if filtered_detections:
                image_detections.append({
                    'image': {
                        'id': fname_id,
                        'path': image_path,
                        'segmented_path': None,
                        'width': image_width,
                        'height': image_height
                    },
                    'detections': filtered_detections
                })
        return image_detections


class UdacityAuttiIngestor(Ingestor):
    def validate(self, root):
        labels_path = f"{root}/labels.csv"
        if not os.path.isfile(labels_path):
            return False, f"Expected to find {labels_path}"
        return True, None

    def ingest(self, root):
        labels_path = f"{root}/labels.csv"
        image_labels = defaultdict(list)

        with open(labels_path) as labels_file:
            labels_csv = csv.reader(labels_file, delimiter=' ')
            next(labels_csv, None)  # skip header
            for idx, row in enumerate(labels_csv):
                image_labels[row[0]].append(row)

        image_detections = []
        for idx, image_path in enumerate(glob.glob(f"{root}/*.jpg")):
            f_name = image_path.split("/")[-1]
            f_image_labels = image_labels[f_name]
            fname_id = f_name.split('.')[0]

            image_width, image_height = _image_dimensions(image_path)

            def clamp_bbox(det):
                if det['right'] > image_width - 1:
                    det['right'] = image_width - 1
                if det['bottom'] > image_height - 1:
                    det['bottom'] = image_height - 1
                return det

            def valid_bbox(det):
                return det['right'] > det['left'] and det['bottom'] > det['top']

            detections = []

            for image_label in f_image_labels:
                x1, y1, x2, y2, x, label = image_label[1:7]
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                detections.append({
                    'label': label,
                    'left': x1,
                    'right': x2,
                    'top': y1,
                    'bottom': y2
                })

            filtered_detections = [clamp_bbox(det) for det in detections if valid_bbox(det)]
            if filtered_detections:
                image_detections.append({
                    'image': {
                        'id': fname_id,
                        'path': image_path,
                        'segmented_path': None,
                        'width': image_width,
                        'height': image_height
                    },
                    'detections': filtered_detections
                })
        return image_detections


def _image_dimensions(path):
    with Image.open(path) as image:
        return image.width, image.height