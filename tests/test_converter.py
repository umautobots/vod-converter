import context  # augment system path to make imports work
from vod_converter import converter


def test_convert_labels():
    assert [{'detections': [
        {'label': 'person'},
        {'label': 'person'},
        {'label': 'person'},
        {'label': 'rhinoZaurus'}
    ]}] == \
           converter.convert_labels(
               image_detections=[
                   {'detections': [
                       {'label': 'Pedestrian'},
                       {'label': 'pedestrian'},
                       {'label': 'Person'},
                       {'label': 'rhinoZaurus'}
                   ]}
               ],
               expected_labels={'person': ['Pedestrian']},
               select_only_known_labels=False,
               filter_images_without_labels=False
           )


def test_select_only_known_labels():
    assert [{'detections': [
        {'label': 'person'},
        {'label': 'person'},
        {'label': 'person'},
    ]}] == \
           converter.convert_labels(
               image_detections=[
                   {'detections': [
                       {'label': 'Pedestrian'},
                       {'label': 'pedestrian'},
                       {'label': 'Person'},
                       {'label': 'rhinoZaurus'}
                   ]}
               ],
               expected_labels={'person': ['Pedestrian']},
               select_only_known_labels=True,
               filter_images_without_labels=False
           )


def test_filter_images_without_labels():
    assert [{'detections': [
        {'label': 'person'},
        {'label': 'person'},
        {'label': 'person'},
    ]}] == \
           converter.convert_labels(
               image_detections=[
                   {'detections': [
                       {'label': 'Pedestrian'},
                       {'label': 'pedestrian'},
                       {'label': 'Person'},
                       {'label': 'rhinoZaurus'}
                   ],
                   },
                   {'detections': [
                       {'label': 'rhinoZaurus'}
                   ]
                   }
               ],
               expected_labels={'person': ['Pedestrian']},
               select_only_known_labels=True,
               filter_images_without_labels=True
           )
