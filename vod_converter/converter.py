"""
Defines the protocol for converting too and from a common data format and executes
the conversion, validating proper conversion along the way.

For a given dataformat, e.g `voc.py`, if you wish to support reading in of your data format, define
an `Ingestor` that can read in data from a path and return an array of data conforming to `IMAGE_DETECTION_SCHEMA`.

If you wish to support data output, define an `Egestor` that, given an array of data of the same form,
can output the data to the filesystem.

See `main.py` for the supported types, and `voc.py` and `kitti.py` for reference.
"""
from jsonschema import validate as raw_validate
from jsonschema.exceptions import ValidationError as SchemaError


def validate_schema(data, schema):
    """Wraps default implementation but accepting tuples as arrays too.

    https://github.com/Julian/jsonschema/issues/148
    """
    return raw_validate(data, schema, types={"array": (list, tuple)})


IMAGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'path': {'type': 'string'},
        'segmented_path': {
            'anyOf': [
                {'type': 'null'},
                {'type': 'string'}
            ]
        },
        'width': {'type': 'integer', 'minimum': 10},
        'height': {'type': 'integer', 'minimum': 10},
    },
    'required': ['id', 'path', 'segmented_path', 'width', 'height']
}


DETECTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'label': {'type': 'string'},
        'top': {'type': 'number', 'minimum': 0},
        'left': {'type': 'number', 'minimum': 0},
        'right': {'type': 'number', 'minimum': 0},
        'bottom': {'type': 'number', 'minimum': 0}
    },
    'required': ['top', 'left', 'right', 'bottom']
}

IMAGE_DETECTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'image': IMAGE_SCHEMA,
        'detections': {
            'type': 'array',
            'items': DETECTION_SCHEMA
        }
    }
}


def validate_image_detections(image_detections):
    for i, image_detection in enumerate(image_detections):
        try:
            validate_schema(image_detection, IMAGE_DETECTION_SCHEMA)
        except SchemaError as se:
            raise Exception(f"at index {i}") from se
        image = image_detection['image']
        for detection in image_detection['detections']:
            if detection['right'] >= image['width'] or detection['bottom'] >= image['height']:
                raise ValueError(f"Image {image} has out of bounds bounding box {detection}")
            if detection['right'] <= detection['left'] or detection['bottom'] <= detection['top']:
                raise ValueError(f"Image {image} has out of zero dimension {detection}")


class Ingestor:
    def validate(self, path):
        """
        Validate that a path contains files / directories expected for a given data format.

        This is where you can provide feedback to the end user if they are attempting to convert from
        your format but have passed you path to a directory that is missing the expected files or directory
        structure.

        :param path: Where the data is stored
        :return: (sucess, error message), e.g (False, "error message") if anything is awry, (True, None) otherwise.
        """
        return True, None

    def ingest(self, path):
        """
        Read in data from the filesytem.
        :param path: '/path/to/data/'
        :return: an array of dicts conforming to `IMAGE_DETECTION_SCHEMA`
        """
        pass


class Egestor:
    def egest(self, *, image_detections, root):
        """
        Output data to the filesystem.

        :param image_detections: an array of dicts conforming to `IMAGE_DETECTION_SCHEMA`
        :param root: '/path/to/output/data/'
        """
        pass


def convert(*, from_path, ingestor, to_path, egestor):
    """
    Converts between data formats, validating that the converted data matches
    `IMAGE_DETECTION_SCHEMA` along the way.

    :param from_path: '/path/to/read/from'
    :param ingestor: `Ingestor` to read in data
    :param to_path: '/path/to/write/to'
    :param egestor: `Egestor` to write out data
    :return: (success, message)
    """
    from_valid, from_msg = ingestor.validate(from_path)

    if not from_valid:
        return from_valid, from_msg

    image_detections = ingestor.ingest(from_path)
    validate_image_detections(image_detections)

    egestor.egest(image_detections=image_detections, root=to_path)
    return True, ''

