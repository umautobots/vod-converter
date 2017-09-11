# Visual Object Dataset converter

Converts between object dataset formats. Requires Python 3.6.

Example: convert from data in [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php) format to
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html) format:

```
$ python3.6 vod_converter/main.py --from kitti --from-path datasets/mydata-kitti --to voc --to-path datasets/mydata-voc
```

See `main.py` for documentation on how to easily plug in additional data formats; you can define a function
that can read in your data into a common format, and it will be then ready to convert to any supported format.

Similarly, you can implement a single function that takes the common format and outputs to the filesystem in
your format and you will be ready to convert from e.g VOC to yours.

Currently support conversion from:

- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php)
- [KITTI tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)
- [Udacity CrowdAI and AUTTI](https://github.com/udacity/self-driving-car/tree/master/annotations)

to:

- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php)

## That 'train.txt' file for KITTI

When reading in KITTI, the script expects a `train.txt` file that isn't part of the original dataset. This is simply a file with the name of each datapoint you wish to capture. [Here's an example with everything in the training set](https://github.com/umautobots/vod-converter/files/1139276/train.txt). You can also create it like so:

```
$ cd datasets/kitti && ls -1 training/image_2 | cut -d. -f1 > train.txt && cd -
$ head datasets/kitti/train.txt
000000
000001
000002
000003
000004
000005
000006
000007
000008
000009
```

## Python2 support

This project is written using features requiring Python3.6+, but there is [a fork](https://github.com/nghiattran/vod-converter) that has been updated to work in Python2 if you need it.

