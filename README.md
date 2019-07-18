# Object detection api

Detect objects in a photo with pretrained `ssd_mobilenet_V1_coco_11_06_2017` model.

## External files

### `frozen_inference_graph.pb`

1. Download [ssd_mobilenet_v1_coco_11_06_2017.tar.gz](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)
2. Unpack
3. Copy `frozen_inference_graph.pb` file to this directory from there.

### `mscoco_label_map.txt`

Redefine [`mscoco_label_map.pbtxt`](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt) as a list of `(id, display_name)`.

## Quick start

### Install environment

If you want to use `virtualenv` like me,

```bash
python3 -m virtualenv venv
source venv/bin/activate
```

and then install other dependencies.

```bash
pip install -r requirements.txt
```

### Inference

```bash
python inference.py image.jpg
```

### Example

```bash
$ python inference.py image.jpg
Detect object from a photo: image.jpg
[('book', 0.3387419), ('bench', 0.11134028), ...]
```
