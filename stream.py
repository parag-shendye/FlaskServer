# app.py
import sys
sys.path.append("./models/research/")
sys.path.append("./Realtime-Action-Recognition/")

from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

from flask import Flask, render_template, Response
import cv2
import sys
import numpy as np
import os

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import utils.lib_images_io as lib_images_io
import utils.lib_plot as lib_plot
import utils.lib_commons as lib_commons
from utils.lib_openpose import SkeletonDetector
from utils.lib_tracker import Tracker
from utils.lib_tracker import Tracker
from utils.lib_classifier import ClassifierOnlineTest
from utils.lib_classifier import *  # Import all sklearn related libraries


app = Flask(__name__)
data = pickle.loads(open("./Facial_Recognition/encodings.pickle","rb").read())

# Pose Estimation Settings
ROOT = "./Realtime-Action-Recognition/"
cfg_all = lib_commons.read_yaml(ROOT+ "config/config.yaml")
cfg = cfg_all["s5_test.py"]

CLASSES = np.array(cfg_all["classes"])

# Action recognition: number of frames used to extract features.
WINDOW_SIZE = int(cfg_all["features"]["window_size"])

# Openpose settings
OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

# Display settings
img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])

@app.route('/')
def index():
    return render_template('index.html')

def get_frame(recognition_methods):
    camera_port=0
    camera=cv2.VideoCapture("./Realtime-Action-Recognition/data/IMG_4560.mp4") #this makes a web cam object
    graph, category_index = init_object_detection()
    skeleton_detector, multiperson_tracker, multiperson_classifier = init_pose_detection()

    while True:
        retval, im = camera.read()
        image = im
        if 'face' in recognition_methods:
            image = recognize_face(image)
        if 'object' in recognition_methods:
            image = recognize_objects(image, graph, category_index)
        if 'pose' in recognition_methods:
            image =  recognize_pose(image, skeleton_detector, multiperson_tracker, multiperson_classifier)
        imgencode=cv2.imencode('.jpg',image)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    del(camera)

def recognize_face(frame):
	# loop over the recognized faces
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
    return frame

def init_object_detection():
    MODEL_NAME = 'inference_graph1'
    PATH_TO_FROZEN_GRAPH = f'./models/research/object_detection/{MODEL_NAME}/frozen_inference_graph.pb'
    PATH_TO_LABELS = './models/research/object_detection/training/labelmap.pbtxt'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    return detection_graph, category_index

def recognize_objects(image_np, detection_graph, category_index):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
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
                    line_thickness=8)
                return image_np

# Pose Estimation Code: 
class MultiPersonClassifier(object):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE, human_id)

    def classify(self, dict_id2skeleton):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)  # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        ''' Get the classifier based on the person id.
        Arguments:
            id {int or "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons


def draw_result_img(img_disp, humans, dict_id2skeleton,dict_id2label,
                    skeleton_detector, multiperson_classifier, scale_h):
    ''' Draw skeletons, labels, and prediction scores onto image for display '''

    # Resize to a proper size for display
    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
    img_disp = cv2.resize(img_disp,
                          dsize=(desired_cols, img_disp_desired_rows))

    # Draw all people's skeleton
    skeleton_detector.draw(img_disp, humans)

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # scale the y data back to original
            skeleton[1::2] = skeleton[1::2] / scale_h
            # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            lib_plot.draw_action_result(img_disp, id, skeleton, "")

    # Add blank to the left for displaying prediction scores of each class
    # img_disp = lib_plot.add_white_region_to_left_of_image(img_disp)

    cv2.putText(img_disp, "Frame:" + str(0),
                (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 0), thickness=2)

    # Draw predicting score for only 1 person
    if len(dict_id2skeleton):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')
        classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp


def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
    '''
    In each image, for each skeleton, save the:
        human_id, label, and the skeleton positions of length 18*2.
    So the total length per row is 2+36=38
    '''
    skels_to_save = []
    for human_id in dict_id2skeleton.keys():
        label = dict_id2label[human_id]
        skeleton = dict_id2skeleton[human_id]
        skels_to_save.append([[human_id, label] + skeleton.tolist()])
    return skels_to_save


def init_pose_detection():
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()
    multiperson_classifier = MultiPersonClassifier("./Realtime-Action-Recognition/model/trained_withFight_classifier.pickle", CLASSES)
    return skeleton_detector, multiperson_tracker, multiperson_classifier

def recognize_pose(img, skeleton_detector, multiperson_tracker, multiperson_classifier):
    # -- Detect skeletons
    humans = skeleton_detector.detect(img)
    skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
    skeletons = remove_skeletons_with_few_joints(skeletons)
    dict_id2label = {}
    # -- Track people
    dict_id2skeleton = multiperson_tracker.track(
        skeletons)  # int id -> np.array() skeleton

    # -- Recognize action of each person
    if len(dict_id2skeleton):
        dict_id2label = multiperson_classifier.classify(dict_id2skeleton)

    # -- Draw
    img_disp = draw_result_img(img, humans, dict_id2skeleton,dict_id2label,
                                skeleton_detector, multiperson_classifier, scale_h)
    return img_disp



@app.route('/calc')
def calc():
     return Response(get_frame(['face','object', 'pose']),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face')
def face():
     return Response(get_frame(['face']),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/obj')
def obj():
     return Response(get_frame(['object']),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose')
def pose():
     return Response(get_frame(['pose']),mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)
