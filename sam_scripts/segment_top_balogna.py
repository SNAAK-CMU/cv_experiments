import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from segment_utils import get_hsv_range, segment_from_hsv, calc_bbox_from_mask, convert_mask_to_orig_dims


# Parameters
CROP_XMIN = 300
CROP_YMIN = 50
CROP_XMAX = 430
CROP_YMAX = 280

CROP_XMIN_HSV = 330
CROP_YMIN_HSV = 100
CROP_XMAX_HSV = 400
CROP_YMAX_HSV = 250

MEAT_W = 100
MEAT_H = 100

LOWER_HSV_WHITE = np.array([0, 0, 66])
UPPER_HSV_WHITE = np.array([176, 53, 255])


def create_sam_predictor():
    # Checkpoint (absolute path to local copy)
    sam2_checkpoint = "/home/snaak/Documents/Abhi/Vision/sam2/checkpoints/sam2.1_hiera_small.pt"
    # Hydra-friendly config name inside the installed package
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    print("SAM predictor created")
    return predictor


def segment_top_balogna(img_name, image, lower_hsv, upper_hsv, predictor):
    # Crop to the region of interest
    crop_xmin = CROP_XMIN
    crop_ymin = CROP_YMIN
    crop_xmax = CROP_XMAX
    crop_ymax = CROP_YMAX
    cropped_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Construct prompt using hsv mask
    mask = segment_from_hsv(cropped_image, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    bounding_box = calc_bbox_from_mask(mask)

    # If HSV produced no bbox, return an empty mask for full image
    if bounding_box is None:
        print(f"No bounding box from HSV mask for {img_name}; skipping SAM on cropped image.")
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Run SAM on cropped region to get all balogna/meat mask
    predictor.set_image(cropped_image)
    try:
        masks, scores, logits = predictor.predict(box=bounding_box[None, :], multimask_output=True)
    except Exception as e:
        print(f"Predictor error for {img_name} on cropped image: {e}")
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    if scores is None or len(scores) == 0:
        print(f"SAM returned no masks for {img_name}")
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    all_meat_mask = masks[np.argmax(scores)]

    # Convert mask back to original image dims and postprocess
    all_meat_mask_orig = convert_mask_to_orig_dims(all_meat_mask, image, crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    kernel = np.ones((5, 5), np.uint8)
    all_meat_mask_orig = cv2.erode(all_meat_mask_orig, kernel, iterations=1)

    # Define expected top-meat dimensions
    meat_w = 100
    meat_h = 100
    y_indices, x_indices = np.where(all_meat_mask_orig * 255 == 255)

    if y_indices.size == 0 or x_indices.size == 0:
        print("No white object found after SAM for", img_name)
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    meat_bottom_centre_x = (xmin + xmax) // 2
    meat_bottom_y = ymax
    meat_top_y = ymax - meat_h
    meat_right_x = meat_bottom_centre_x + (meat_w // 2)
    meat_left_x = meat_bottom_centre_x - (meat_w // 2)
    meat_box = np.array([meat_left_x, meat_top_y, meat_right_x, meat_bottom_y])

    # Run SAM on the top meat box (on the full image)
    predictor.set_image(image)
    try:
        masks, scores, logits = predictor.predict(box=meat_box[None, :], multimask_output=True)
    except Exception as e:
        print(f"Predictor error for {img_name} on meat box: {e}")
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    if scores is None or len(scores) == 0:
        print(f"SAM returned no masks for top meat box for {img_name}")
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    top_meat_mask = masks[np.argmax(scores)]
    kernel = np.ones((7, 7), np.uint8)
    top_meat_mask = cv2.erode(top_meat_mask, kernel, iterations=1)

    return top_meat_mask


def test_on_dir(dir_path):
    crop_xmin_hsv = CROP_XMIN_HSV
    crop_ymin_hsv = CROP_YMIN_HSV
    crop_xmax_hsv = CROP_XMAX_HSV
    crop_ymax_hsv = CROP_YMAX_HSV

    first_img = True
    lower_hsv = None
    upper_hsv = None

    predictor = create_sam_predictor()

    # Recursive walk and process images, mirroring folder structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(script_dir, "sam_tests", "bologna_results")
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for root, _, files in os.walk(dir_path):
        for img_name in sorted(files):
            if not img_name.lower().endswith(supported_exts):
                continue

            img_path = os.path.join(root, img_name)
            print("Processing", img_path)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: failed to read image {img_path}, skipping")
                continue

            if first_img:
                first_img = False
                try:
                    cropped2_image = image[crop_ymin_hsv:crop_ymax_hsv, crop_xmin_hsv:crop_xmax_hsv]
                    lower_hsv, upper_hsv = get_hsv_range(cropped2_image)
                except Exception as e:
                    print(f"Error computing HSV range from first image {img_path}: {e}")
                    continue

            top_meat_mask = segment_top_balogna(img_name, image, lower_hsv, upper_hsv, predictor)
            top_meat_mask *= 255

            color = (0, 0, 255)
            colored_image = np.zeros_like(image)
            colored_image[:] = color
            result = np.where(top_meat_mask[:, :, None] == 255, colored_image, image)

            rel_path = os.path.relpath(img_path, dir_path)
            rel_dir = os.path.dirname(rel_path)
            out_dir = os.path.join(output_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.splitext(os.path.basename(img_name))[0] + "_result.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, result)


if __name__ == "__main__":
    # Replace this with your dataset path as needed
    test_on_dir("/home/snaak/Documents/datasets/testsets/meats")
