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

from converter import Ingestor, Egestor

# Detections

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
        image_names = self._get_image_ids(path)
        return [self._get_image_detection(path, image_name) for image_name in image_names]

    def _get_image_ids(self, root):
        path = f"{root}/train.txt"
        with open(path) as f:
            return f.read().strip().split('\n')


    def _get_image_detection(self, root, image_id):
        detections_fpath = f"{root}/training/label_2/{image_id}.txt"
        detections = self._get_detections(detections_fpath)
        detections = [det for det in detections if det['left'] < det['right'] and det['top'] < det['bottom']]
        return {
            'image': {
                'id': image_id,
                'path': f"{root}/training/image_2/{image_id}.png",
                'segmented_path': None,
                'width': 1024,
                'height': 512
                # 'width': 1242,
                # 'height': 375
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
                    'left': max(0.0, x1),
                    'right': min(1023, x2),
                    'top': max(0.0, y1),
                    'bottom': min(511, y2)
                })
        return detections


# TODO
# class KITTIEgestor(Egestor):
#     pass


