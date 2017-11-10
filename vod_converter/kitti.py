"""
Ingestor for KITTI formats.

http://www.cvlibs.net/datasets/kitti/eval_object.php

Per devkit docs:

All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.


"""

import csv
import os
from PIL import Image
import shutil

from converter import Ingestor, Egestor


class KITTIIngestor(Ingestor):
    def validate(self, path):
        expected_dirs = [
            'training/image_2',
            'training/label_2'
        ]
        for subdir in expected_dirs:
            if not os.path.isdir(f"{path}/{subdir}"):
                return False, f"Expected subdirectory {subdir} within {path}"
        if not os.path.isfile(f"{path}/train.txt"):
            return False, f"Expected train.txt file within {path}"
        return True, None

    def ingest(self, path):
        image_ids = self._get_image_ids(path)
        image_ext = 'png'
        if len(image_ids):
            first_image_id = image_ids[0]
            image_ext = self.find_image_ext(path, first_image_id)
        return [self._get_image_detection(path, image_name, image_ext=image_ext) for image_name in image_ids]

    def find_image_ext(self, root, image_id):
        for image_ext in ['png', 'jpg']:
            if os.path.exists(f"{root}/training/image_2/{image_id}.{image_ext}"):
                return image_ext
        raise Exception(f"could not find jpg or png for {image_id} at {root}/training/image_2")

    def _get_image_ids(self, root):
        path = f"{root}/train.txt"
        with open(path) as f:
            return f.read().strip().split('\n')

    def _get_image_detection(self, root, image_id, *, image_ext='png'):
        detections_fpath = f"{root}/training/label_2/{image_id}.txt"
        detections = self._get_detections(detections_fpath)
        detections = [det for det in detections if det['left'] < det['right'] and det['top'] < det['bottom']]
        image_path = f"{root}/training/image_2/{image_id}.{image_ext}"
        image_width, image_height = _image_dimensions(image_path)
        return {
            'image': {
                'id': image_id,
                'path': image_path,
                'segmented_path': None,
                'width': image_width,
                'height': image_height
            },
            'detections': detections
        }

    def _get_detections(self, detections_fpath):
        detections = []
        with open(detections_fpath) as f:
            f_csv = csv.reader(f, delimiter=' ')
            for row in f_csv:
                x1, y1, x2, y2 = map(float, row[4:8])
                label = row[0]
                detections.append({
                    'label': label,
                    'left': x1,
                    'right': x2,
                    'top': y1,
                    'bottom': y2
                })
        return detections


def _image_dimensions(path):
    with Image.open(path) as image:
        return image.width, image.height

DEFAULT_TRUNCATED = 0.0 # 0% truncated
DEFAULT_OCCLUDED = 0    # fully visible

class KITTIEgestor(Egestor):

    def expected_labels(self):
        return {
            'Car': [],
            'Cyclist': ['biker'],
            'Misc': [],
            'Pedestrian': ['person'],
            'Person_sitting': [],
            'Tram': [],
            'Truck': [],
            'Van': [],
        }

    def egest(self, *, image_detections, root):
        images_dir = f"{root}/training/image_2"
        os.makedirs(images_dir, exist_ok=True)
        labels_dir = f"{root}/training/label_2"
        os.makedirs(labels_dir, exist_ok=True)

        id_file = f"{root}/train.txt"

        for image_detection in image_detections:
            image = image_detection['image']
            image_id = image['id']
            src_extension = image['path'].split('.')[-1]
            shutil.copyfile(image['path'], f"{images_dir}/{image_id}.{src_extension}")

            with open(id_file, 'a') as out_image_index_file:
                out_image_index_file.write(f'{image_id}\n')

            out_labels_path = f"{labels_dir}/{image_id}.txt"
            with open(out_labels_path, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)

                for detection in image_detection['detections']:
                    kitti_row = [-1] * 15
                    kitti_row[0] = detection['label']
                    kitti_row[1] = DEFAULT_TRUNCATED
                    kitti_row[2] = DEFAULT_OCCLUDED
                    x1 = detection['left']
                    x2 = detection['right']
                    y1 = detection['top']
                    y2 = detection['bottom']
                    kitti_row[4:8] = x1, y1, x2, y2
                    csvwriter.writerow(kitti_row)



