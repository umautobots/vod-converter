# Visual Object Dataset converter

Converts between object dataset formats. Requires Python 3.6.

Example: convert from data in [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php) format to
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html) format:

```
$ python3.6 vod_converter/main.py --from kitti --from-path datasets/mydata-kitti --to voc --to-path datasets/mydata-voc
```

See `main.py` for documentation on how to easily plug in additional data formats; you can define a function
that can read in your data into a common format, and it will be then ready to convert to any supported format.