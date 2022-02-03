import tensorflow as tf
from PIL import Image
from io import BytesIO
import time
import numpy as np


def preprocess_image(file, max_size=(1028, 1028)):
    image_raw = Image.open(BytesIO(file)).convert('RGB')
    image_raw.thumbnail(max_size, Image.ANTIALIAS) # rescale image to be smaller than max size
    # image_raw = tf.keras.preprocessing.image.img_to_array(image_raw)
    image_numpy = np.array(image_raw)
    img_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)
    converted_img  = tf.image.convert_image_dtype(img_tensor, tf.float32)[tf.newaxis, ...]
    return converted_img


def run_detector(detector, file):
    img = preprocess_image(file)

    start_time = time.time()
    result = detector(img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time-start_time)
    print(result['detection_class_entities'])
    return result



def filter_bees(result: dict, min_score: float = 0.3):

    print(f"Total {len(result['detection_boxes'])} objects detected.")
    # filter by only bee labels
    bee_idx = [idx for idx, ele in enumerate(result["detection_class_entities"]) if ele == b"Bee"]
    print(f"Among them, a total of {len(bee_idx)} are classified as bee.")
    
    # filter by high confidence score
    bee_idx_high_conf = [ele for ele in bee_idx if (result["detection_scores"][ele] >= min_score)]
    print(f"Among them, a total of {len(bee_idx_high_conf)} have prediction confidence score of at least {min_score}.")
    
    # apply filter
    result_bees = {'detection_scores':[],
                  'detection_boxes':[], 
                  # 'detection_class_entities':[],
                  }
    
    for idx in bee_idx_high_conf:
        # result_bees['detection_class_entities'].append(result['detection_class_entities'][idx]) 
        result_bees['detection_scores'].append(float(result['detection_scores'][idx]))
        result_bees['detection_boxes'].append(result['detection_boxes'][idx].tolist()) 
    return result_bees