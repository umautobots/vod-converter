"""
Converts between visual object detection dataset formats. See `converter.py` for more info.

To add support for additional data formats, define a module with an `converter.Ingestor` and/or
`converter.Egestor` implementation and add them to the `INGESTORS` and `EGESTORS` dicts below.
"""

import argparse
import logging

import converter
import kitti
import kitti_tracking
import voc

import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)

INGESTORS = {
    'kitti': kitti.KITTIIngestor(),
    'kitti-tracking': kitti_tracking.KITTITrackingIngestor(),
    'voc': voc.VOCIngestor()
}

EGESTORS = {
    'voc': voc.VOCEgestor(),
    # 'kitti': kitti.KITTIEgestor() # TODO
}


def main(*, from_path, from_key, to_path, to_key):
    success, msg = converter.convert(from_path=from_path, ingestor=INGESTORS[from_key],
                                     to_path=to_path, egestor=EGESTORS[to_key])
    if success:
        print(f"Successfully converted from {from_key} to {to_key}.")
    else:
        print(f"Failed to convert from {from_key} to {to_key}: {msg}")
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description='Convert visual object datasets.')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('--from',
                          dest='from_key',
                          required=True,
                          help=f'Format to convert from: one of {", ".join(INGESTORS.keys())}', type=str)
    required.add_argument('--from-path', dest='from_path',
                          required=True,
                          help=f'Path to dataset you wish to convert.', type=str)
    required.add_argument('--to', dest='to_key', required=True,
                          help=f'Format to convert to: one of {", ".join(EGESTORS.keys())}',
                          type=str)
    optional.add_argument(
        '--to-path',
        dest='to_path',
        help=f'Path to output directory for converted datset. If omitted, one will be created based on your '
             f'input directory and output format, e.g "/path/to/dataset-voc"', type=str, default=None)
    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(from_path=args.from_path, from_key=args.from_key, to_path=args.to_path, to_key=args.to_key))
