import tensorflow as tf
from PIL import Image
from io import BytesIO
import time




def preprocess_and_save_input_image(img, input_filename, max_size=(1028, 1028)):
    image_raw = Image.open(BytesIO(img)).convert('RGB')
    image_raw.thumbnail(max_size, Image.ANTIALIAS) # rescale image to be smaller than max size
    image_raw.save(input_filename)




def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img




def run_detector(detector, path: str):
    img = load_img(path)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time-start_time)

    return result



def filter_bees(result: dict, min_score: float = 0.2):

    print(f"Total {len(result['detection_boxes'])} objects detected.")
    # filter by only bees or insect labels
    bee_idx = [idx for idx, ele in enumerate(result["detection_class_entities"]) if ele == b"Bee" or ele == b"Insect"]
    print(f"Among them, a total of {len(bee_idx)} are classified as bee or insect.")
    
    # filter by high confidence score
    bee_idx_high_conf = [ele for ele in bee_idx if (result["detection_scores"][ele] >= min_score)]
    print(f"Among them, a total of {len(bee_idx_high_conf)} have prediction confidence score of at least {min_score}.")
    
    # apply filter
    result_bees = {'detection_class_entities':[],
                  'detection_scores':[],
                  'detection_boxes':[]}
    
    for idx in bee_idx_high_conf:
        result_bees['detection_class_entities'].append(result['detection_class_entities'][idx]) 
        result_bees['detection_scores'].append(float(result['detection_scores'][idx]))
        result_bees['detection_boxes'].append(result['detection_boxes'][idx].tolist()) 
    return result_bees