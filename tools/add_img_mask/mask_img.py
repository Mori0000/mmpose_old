from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import os

import json
import torch
from typing import List


# jsonファイルへの処理
def traverse_json(data, level=0):
    indent = "  " * level
    if isinstance(data, dict):
        for key, value in data.items():
            traverse_json(value, level + 1)
    elif isinstance(data, list):
        if len(data) > 0:
            traverse_json(data[0], level + 1)


def print_json_structure(json_content):
    data = json.loads(json_content)
    traverse_json(data)


def get_info_by_id(json_data, specified_id):
    image_info = next((item for item in json_data["images"] if item["id"] == specified_id), None)
    annotations_info = [item for item in json_data["annotations"] if item["image_id"] == specified_id]
    return image_info, annotations_info


def print_non_zero_keypoints(annotation_info):
    '''
    keypointのGTの中で、visibilityが0でないもの（画像から見えているもの）のみを抽出する
    '''
    key_points = []
    for i, annotation in enumerate(annotation_info):
        keypoints = annotation['keypoints']
        for j in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[j], keypoints[j+1], keypoints[j+2]
            if visibility != 0:
                key_points.append((x, y)) 
                # break   # このbreakiいれると、一人の人物に対し一点のみを抽出
    return key_points


def detect_faces_centers(specified_id: int) -> List[Tuple[int, int]]:
    '''
    顔画像の中心座標を取得する予定
    現状、annotationのkeypointのうち、visibilityが0でないものを抽出している
    '''
    json_path = '/home/moriki/PoseEstimation/CID/data/crowdpose/annotations/crowdpose_test.json'

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    image_info, annotation_info = get_info_by_id(json_data, specified_id)
    # print(image_info)
    # print(annotation_info)
    # print("")
    key_points = print_non_zero_keypoints(annotation_info)
    return key_points


def apply_mask(image: np.ndarray, mask: np.ndarray, option: str = 'blur', alpha: float = 0.8, blur_strength: int = 5) -> np.ndarray:
    rgb_mask = np.stack([mask]*3, axis=-1)
    if option == 'blur':
        inverted_mask = np.stack([~mask]*3, axis=-1)
        blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        masked_image = np.where(inverted_mask, blurred_image, image)
    elif option == 'clear':
        masked_image = image.copy()
        masked_image[~rgb_mask] = (masked_image[~rgb_mask] * alpha).astype(image.dtype)
    else:  # Assumes a simple binary mask operation if neither 'blur' nor 'clear'
        masked_image = np.where(rgb_mask, image, 0)
    return masked_image


def show_masked_image(image: np.ndarray, mask: np.ndarray, ax: Axes) -> None:
    masked_image = apply_mask(image, mask)
    ax.imshow(masked_image)


def overlay_masks(image: np.ndarray, masks: List[np.ndarray], apply_effect: bool = False, effect_option: str = 'blur') -> np.ndarray:
    combined_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined_mask |= mask
    if apply_effect:
        overlay_image = apply_mask(image, combined_mask, option=effect_option)
    else:
        rgb_mask = np.stack([combined_mask] * 3, axis=-1)
        overlay_image = np.zeros_like(image)
        overlay_image[rgb_mask] = image[rgb_mask]
    return overlay_image


def marking_keypoints(image: np.ndarray, key_points: List[Tuple[int, int]]) -> np.ndarray:
    for (x, y) in key_points:
        cv2.circle(image, (x, y), radius=8, color=(0, 255, 0), thickness=-1)
    return image


def select_largest_mask(masks):
    mask_sizes = masks.sum(axis=(1, 2))
    largest_mask_index = mask_sizes.argmax()
    return masks[largest_mask_index]



key_points = [(217, 189), (269, 220), (272, 286), (267, 318), (149, 251), (196, 265), (184, 330), (230, 333), (183, 405), (225, 421), (286, 187), (253, 203), (218, 157), (245, 180), (152, 201), (166, 208), (289, 194), (245, 171), (411, 199), (413, 186), (450, 247), (453, 250), (451, 302), (402, 267), (489, 277), (503, 264), (452, 338), (477, 339), (440, 424), (464, 417), (356, 184), (397, 195)]


def main():
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = "/home/moriki/PoseEstimation/CID/segment-anything/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:1"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)     # SAMモデルの読み込み
    sam.to(device=device)                                   # デバイスの設定
    predictor = SamPredictor(sam)                           # SAMの予測器を作成

    numbers = [100001]
    effect_option = 'black'

    for num in numbers:
        output_dir = f'/home/moriki/PoseEstimation/CID/segment-anything/example_output/'
        os.makedirs(output_dir, exist_ok=True)
        image_path = f"/home/moriki/PoseEstimation/CID/data/crowdpose/images/{num}.jpg"
        image_origin = cv2.imread(image_path)
        save_dir = os.path.join(output_dir, f"{num}")
        os.makedirs(save_dir, exist_ok=True)
        
        if image_origin is None:
            continue  # Skip if the image couldn't be read
        cv2.imwrite(os.path.join(save_dir, 'original_image.png'), image_origin)  # 画像をそのままBGR形式で保存

        image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        centers = key_points
        marked_image = marking_keypoints(image_origin.copy(), centers)
        cv2.imwrite(os.path.join(save_dir, 'marked_image.png'), marked_image)  # 画像をそのままBGR形式で保存
        predictor.set_image(image)
        
        overlaied_masks = []
        num_center = 1
        for center in centers:
            center = [center]
            input_points = np.array(center)
            input_labels = np.ones(len(center))  # すべての点に対して同一のラベルを設定
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            mask = select_largest_mask(masks)
            overlaied_masks.append(mask)
            mask_image = apply_mask(image=image, mask=mask, option=effect_option)  # マスクを画像に適用
            plt.figure(figsize=(10, 10))
            plt.imshow(mask_image)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'1{num:05}_{num_center}.jpg'), bbox_inches='tight', pad_inches=0)
            plt.close()
            num_center += 1

        overlayed_image = overlay_masks(image=image, masks=overlaied_masks, apply_effect=True, effect_option=effect_option)
        plt.imshow(overlayed_image)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'1{num:05}.jpg'), bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    main()