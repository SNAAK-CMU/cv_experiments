import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from segment_utils import get_hsv_range, segment_from_hsv, calc_bbox_from_mask, convert_mask_to_orig_dims


# Parameters
CROP_XMIN_HSV = 350
CROP_YMIN_HSV = 100
CROP_XMAX_HSV = 400
CROP_YMAX_HSV = 250

CROP_XMIN = 320
CROP_YMIN = 60
CROP_XMAX = 430
CROP_YMAX = 280



def create_sam_predictor():
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def segment_top_cheese_2e2(img_name, image, lower_hsv, upper_hsv, predictor):
    # Crop out the inner borders of the bin
    crop_xmin = CROP_XMIN
    crop_ymin = CROP_YMIN
    crop_xmax = CROP_XMAX
    crop_ymax = CROP_YMAX
    cropped_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    
    # Contruct prompt using hsv mask
    mask = segment_from_hsv(cropped_image, lower_hsv, upper_hsv)
    # cv2.imwrite(f"{img_name}_hsvmask.jpg", mask)
    bounding_box = calc_bbox_from_mask(mask)

    # Run SAM
    predictor.set_image(cropped_image)
    masks, scores, logits = predictor.predict(box=bounding_box[None, :], multimask_output=True)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    all_cheese_mask = masks[np.argmax(scores)]

    # Process mask from SAM
    all_cheese_mask_orig = convert_mask_to_orig_dims(all_cheese_mask, image, crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel
    all_cheese_mask_orig = cv2.erode(all_cheese_mask_orig, kernel, iterations=1)
    

    # Get top cheese box
    # Cheese dims in pix
    cheese_w = 90
    cheese_h = 90
    y_indices, x_indices = np.where(all_cheese_mask_orig * 255 == 255)

    # Ensure there are white pixels
    if y_indices.size == 0 or x_indices.size == 0:
        print("No white object found!")

    # Find bounding box of the vertical rectangle
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    cheese_bottom_centre_x = (xmin + xmax) // 2
    cheese_bottom_y = ymax
    cheese_top_y = ymax - cheese_h
    cheese_right_x = cheese_bottom_centre_x + (cheese_w // 2)
    cheese_left_x = cheese_bottom_centre_x - (cheese_w // 2)
    cheese_box = np.array([cheese_left_x, cheese_top_y, cheese_right_x, cheese_bottom_y])
    
    # Run SAM on top cheese box
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(box=cheese_box[None, :], multimask_output=True)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    top_cheese_mask = masks[np.argmax(scores)]
    kernel = np.ones((7, 7), np.uint8)
    top_cheese_mask = cv2.erode(top_cheese_mask, kernel, iterations=1)

    return top_cheese_mask

def test_on_dir(dir_path):
    crop_xmin_hsv = CROP_XMIN_HSV
    crop_ymin_hsv = CROP_YMIN_HSV
    crop_xmax_hsv = CROP_XMAX_HSV
    crop_ymax_hsv = CROP_YMAX_HSV

    first_img = True
    lower_hsv = None
    upper_hsv = None

    predictor = create_sam_predictor()

    image_names = sorted(os.listdir(dir_path))
    for img_name in image_names:

        img_path = os.path.join(dir_path, img_name)
        image = cv2.imread(img_path)

        if first_img:
            first_img = False
            cropped2_image = image[crop_ymin_hsv:crop_ymax_hsv, crop_xmin_hsv:crop_xmax_hsv]
            lower_hsv, upper_hsv = get_hsv_range(cropped2_image)

        top_cheese_mask = segment_top_cheese_2e2(img_name, image, lower_hsv, upper_hsv, predictor)
        top_cheese_mask *= 255

        color = (0, 0, 255)
        colored_image = np.zeros_like(image)
        colored_image[:] = color
        result = np.where(top_cheese_mask[:, :, None] == 255, colored_image, image)

        cv2.imwrite(f"../../data_collection/results/{img_name.split('.')[0]}_result.jpg", result)

if __name__ == "__main__":
    test_on_dir("../../data_collection/cheese_sequence")