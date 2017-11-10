"""
Ingestor for KITTI tracking formats.

http://www.cvlibs.net/datasets/kitti/eval_tracking.php

Note: even though this is for tracking instead of object detection, sometime it's helpful to convert
data from this for object detection training. This reads in the left color labels.

Per devkit docs:

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/%04d/ contains the left color camera sequence images (png)
  - image_03/%04d/ contains the right color camera sequence images  (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text files)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m) provided within this devkit.
All values (numerical or strings) are separated via spaces, each row
corresponds to one object. The 17 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries.
		     Truncation 2 indicates an ignored object (in particular
		     in the beginning or end of a track) introduced by manual
		     labeling.
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
from collections import defaultdict
import os
import re
from PIL import Image

from converter import Ingestor

LABEL_F_PATTERN = re.compile('[0-9]+\.txt')


class KITTITrackingIngestor(Ingestor):
    def validate(self, path):
        expected_dirs = [
            'image_02',
            'label_02'
        ]
        for subdir in expected_dirs:
            if not os.path.isdir(f"{path}/{subdir}"):
                return False, f"Expected subdirectory {subdir} within {path}"
        return True, None

    def ingest(self, path):
        fs = os.listdir(f"{path}/label_02")
        label_fnames = [f for f in fs if LABEL_F_PATTERN.match(f)]
        image_detections = []
        for label_fname in label_fnames:
            frame_name = label_fname.split(".")[0]
            labels_path = f"{path}/label_02/{label_fname}"
            images_dir = f"{path}/image_02/{frame_name}"
            image_detections.extend(
                self._get_track_image_detections(frame_name=frame_name, labels_path=labels_path, images_dir=images_dir))
        return image_detections

    def _get_track_image_detections(self, *, frame_name, labels_path, images_dir):
        detections_by_frame = defaultdict(list)
        with open(labels_path) as f:
            f_csv = csv.reader(f, delimiter=' ')
            for row in f_csv:
                frame_id = int(row[0])
                x1, y1, x2, y2 = map(float, row[6:10])
                label = row[2]
                detections_by_frame[frame_id].append({
                    'label': label,
                    'left': x1,
                    'right': x2,
                    'top': y1,
                    'bottom': y2
                })

        image_detections = []
        for frame_id in sorted(detections_by_frame.keys()):
            frame_dets = detections_by_frame[frame_id]
            image_path = f"{images_dir}/{frame_id:06d}.png"
            if not os.path.exists(image_path):
                image_path = f"{images_dir}/{frame_id:06d}.jpg"
            with Image.open(image_path) as image:
                image_width = image.width
                image_height = image.height

                def clamp_bbox(det):
                    if det['right'] > image_width - 1:
                        det['right'] = image_width - 1
                    if det['bottom'] > image_height - 1:
                        det['bottom'] = image_height - 1
                    return det

                image_detections.append({
                    'image': {
                        'id': f"{frame_name}-{frame_id:06d}",
                        'path': image_path,
                        'segmented_path': None,
                        'width': image.width,
                        'height': image.height
                    },
                    'detections': [clamp_bbox(det) for det in frame_dets]
                })
        return image_detections
