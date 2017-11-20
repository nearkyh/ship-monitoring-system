import tensorflow as tf
import numpy as np
import os
import sys
import time
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def gen():
  i=1
  while i<10:
    yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
    i+=1

def get_frame():
  video = sys.argv[1]  # video file path, ex) *.mp4
  # video = "rtsp://192.168.0.128:8091/test1.mp4"  # To artik710
  # video = "rtsp://192.168.0.145:8091/test1.mp4"  # To raspberryPi3

  cap = cv2.VideoCapture(video)

  # This is needed since the notebook is stored in the object_detection folder.
  sys.path.append("..")

  # Object detection imports
  # Here are the imports from the object detection module.
  from object_detection.utils import label_map_util
  from object_detection.utils import visualization_utils as vis_util

  # Model preparation
  MODEL_NAME = 'object_detection/export_models/inference_graph_rfcn_resnet101_30000'

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = os.path.join('object_detection/data', 'object-detection.pbtxt')

  NUM_CLASSES = 1

  # Load a (frozen) Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Loading label map
  # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                              use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  # Detection
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      prevTime = 0  # Frame time variable
      i=1
      while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=.5,
          line_thickness=2)

        ################### Data analysis ###################
        final_score = np.squeeze(scores)  # scores
        r_count = 0  # counting
        r_score = []  # temp score, <class 'numpy.ndarray'>
        final_category = np.array([category_index.get(i) for i in classes[0]])  # category
        r_category = np.array([])  # temp category

        for i in range(100):
          if scores is None or final_score[i] > 0.5:
            r_count = r_count + 1
            r_score = np.append(r_score, final_score[i])
            r_category = np.append(r_category, final_category[i])

        if r_count > 0:
          for i in range(len(r_score)):  # socre array`s length
            data = "Object Num: {} || Category: {} || Score: {}%".format(i + 1, r_category[i]['name'], 100 * r_score[i])
            cv2.putText(image_np, data, (5, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            final_boxes = np.squeeze(boxes)[i]  # ymin, xmin, ymax, xmax
            xmin = final_boxes[1]
            ymin = final_boxes[0]
            xmax = final_boxes[3]
            ymax = final_boxes[2]
            location_x = (xmax + xmin) / 2
            location_y = (ymax + ymin) / 2
            data2 = "Location (x: {}, y: {})".format(location_x, location_y)
            cv2.putText(image_np, "Location (x: {}, y: {})".format(location_x, location_y), (5, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        else:
          cv2.putText(image_np, "Not Detect", (5, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        #####################################################

        # Frame
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        str = "FPS : %0.1f" % fps
        cv2.putText(image_np, str, (5, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        # Trained Model Name
        model_name = MODEL_NAME.split('/')[2]
        cv2.putText(image_np, model_name, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        imgencode=cv2.imencode('.jpg',image_np)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        i+=1
      del(cap)

