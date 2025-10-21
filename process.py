from typing import Tuple, Union
import math
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
import json
from numpy import dot
from numpy.linalg import norm
import tensorflow_hub as hub
# import tf_keras as keras
# from tf_keras import layers
# import tf_keras as keras
# from tensorflow.keras import layers, models
# from google.colab.patches import cv2_imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    # cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    coordinates = {}
    parts = ["right eye", "left eye", "nose tipe", "mouth", "right ear", "left ear"]
    for keypoint, part in zip(detection.keypoints,parts):
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,width, height)
      coordinates[part] = keypoint_px
  return annotated_image, coordinates, start_point, end_point

model_path = "blaze_face_short_range (1).tflite"

def straighten_eyeline(image, left_eye, right_eye):
    # Ensure eyes are numpy arrays
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
        # Compute differences
        delta = right_eye - left_eye
        angle = -np.degrees(np.arctan2(delta[1], delta[0]))
        # Midpoint between eyes
        eye_center = (int(left_eye[0] + right_eye[0])//2, int(left_eye[1] + right_eye[1])//2)
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center = eye_center, angle = angle, scale = 1.0)
        # Rotate
        (h, w) = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h))
        # Rotated image
        return rotated, left_eye, right_eye


def crop_and_resize(image, x1,y1,x2,y2, target_size=(160, 160)):
    # Crop the image
    cropped = image[y1:y2, x1:x2]

    # Resize to target size
    resized  = cv2.resize(cropped, target_size)

    return resized

def build_facenet_model(input_shape=(160, 160, 3), embedding_dim=128):
    # Base CNN for feature extraction
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Global pooling + embedding
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)  # ✅ fixed activation typo
    embedding = tf.keras.layers.Dense(embedding_dim)(x)
    embedding = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding)  # Normalize embeddings

    # Build model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=embedding)
    return model


def preprocess_image(username,image_path):
  # STEP 1: Create a FaceDetector object.
  base_options = python.BaseOptions(model_asset_path=model_path,delegate=python.BaseOptions.Delegate.CPU)
  options = vision.FaceDetectorOptions(base_options=base_options)
  detector = vision.FaceDetector.create_from_options(options)

  # STEP 2: Load the input image.
  image = mp.Image.create_from_file(image_path)
  # cv2_imshow(image.numpy_view())

  # STEP 3: Detect faces in the input image
  detection_result = detector.detect(image)

  # STEP 4: Process the detection result. In this case, visualize it.
  image_copy = np.copy(image.numpy_view())
  annotated_image, coordinates, start_point, end_point = visualize(image_copy, detection_result)
  rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  
  # STEP 5: Using both eyes align the face to be straight
  np_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
  rotated_image, left_eye, right_eye = straighten_eyeline(np_image, np.array(list(coordinates["left eye"])), np.array(list(coordinates["right eye"])))

  # STEP 6: Get Face borderline coordinates
  x1,y1 = start_point[0], start_point[1]
  x2, y2 = end_point[0], end_point[1]
  x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  

  # STEP 7: Crop Face using Border coordinates
  resized_image = crop_and_resize(rotated_image, x1, y1, x2, y2)

  # STEP 8: Convert image to float32
  resized_image = np.expand_dims(resized_image, axis=0).astype('float32')

  # STEP 9: Build Model
  model = build_facenet_model()
  # model.save("face_recognition_model2.h5")
  # model = keras.models.load_model("face_recognition_model.h5",custom_objects={"KerasLayer":hub.KerasLayer})


  # STEP 10: Get Image Array
  image_array = model.predict(resized_image)
  return username, image_array


def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# facenet_model = build_facenet_model()
# facenet_model.save("face_recognition_model2.h5")