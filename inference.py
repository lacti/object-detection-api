import numpy as np
import tensorflow as tf
from PIL import Image


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_label_map():
    with open('mscoco_label_map.txt', 'r') as f:
        text = f.read().strip()
    labels = {}
    for pair in [line.split(' ') for line in text.split('\n')]:
        labels[int(pair[0])] = pair[1]
    return labels


_labels = load_label_map()


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    return output_dict['num_detections'][0], output_dict['detection_classes'][0].astype(np.int64), \
        output_dict['detection_scores'][0], output_dict['detection_boxes'][0]


def load_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('./frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


_graph = load_graph()


def inference_image(image_path):
    global _graph
    global _labels
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    _, classes, scores, _ = run_inference_for_single_image(
        image_np_expanded, _graph)
    return list(zip([_labels[c] for c in classes], scores))


if __name__ == '__main__':
    import sys

    print('Detect object from a photo:', sys.argv[1])
    print(inference_image(sys.argv[1]))
