# For downloading the image.
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # don't use GUI

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def report_image(full_filename, obj_result, health_result):
    img = Image.open(full_filename)
    img.thumbnail((512,512), Image.ANTIALIAS) # rescale image to be smaller than max size
    image_with_boxes = draw_boxes(
    np.array(img), np.array(obj_result["detection_boxes"]), obj_result["detection_scores"], health_result)

    dest_fp = save_image(image_with_boxes, full_filename.split('/')[-1])
    return dest_fp


##### helper visualization functions 

def draw_boxes(image, boxes, scores, health_result, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""

    font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(health_result['label'][i],
                                         int(100 * health_result['confidence'][i]))
            if health_result['label'][i] == 'healthy':
                color = 'green'
            else:
                color = 'red'

            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
              image_pil,
              ymin,
              xmin,
              ymax,
              xmax,
              color,
              font,
              display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


def save_image(image, image_filename):
    dest_fp = f"web_app/static/img/results/{image_filename}"

    plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(dest_fp, bbox_inches='tight', pad_inches = 0)

    return dest_fp


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin