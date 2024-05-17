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
    # breakpoint()
    image_info = next((item for item in json_data["images"] if int(item["id"]) == specified_id), None)
    annotations_info = [item for item in json_data["annotations"] if int(item["image_id"]) == specified_id]
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
                # break   # このbreakいれると、一人の人物に対し一点のみを抽出
    return key_points


def detect_faces_centers(json_path: str, specified_id: int) -> List[Tuple[int, int]]:
    '''
    顔画像の中心座標を取得する予定だった
    現状、annotationのkeypointのうち、visibilityが0でないものを抽出している
    '''

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    image_info, annotation_info = get_info_by_id(json_data, specified_id)
    key_points = print_non_zero_keypoints(annotation_info)
    return key_points


import cv2
import numpy as np

def apply_mask(image: np.ndarray, mask: np.ndarray, option: str, alpha: float, blur_strength: int, block_size: int) -> np.ndarray:
    rgb_mask = np.stack([mask]*3, axis=-1)
    if option == 'blur':
        inverted_mask = np.stack([~mask]*3, axis=-1)
        blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        masked_image = np.where(inverted_mask, blurred_image, image)
    elif option == 'clear':
        masked_image = image.copy()
        masked_image[~rgb_mask] = (masked_image[~rgb_mask] * alpha).astype(image.dtype)
    elif option == 'mosaic':
        inverted_mask = ~mask  # マスクを反転
        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                x_start, x_end = i, min(i + block_size, image.shape[0])
                y_start, y_end = j, min(j + block_size, image.shape[1])
                if np.any(inverted_mask[x_start:x_end, y_start:y_end]):
                    color = image[x_start:x_end, y_start:y_end].mean(axis=(0, 1)).astype(int)
                    image[x_start:x_end, y_start:y_end] = color
        masked_image = image
    elif option == 'checkerboard':
        # breakpoint()
        grid_size_h = image.shape[0] // 200
        grid_size_w = image.shape[1] // 200
        inverted_mask = ~mask  # マスクを反転して外側のみを処理
        for i in range(200):
            for j in range(200):
                if (i + j) % 2 == 0:  # 1マスおきに黒く塗りつぶし
                    x_start = i * grid_size_h
                    x_end = min((i + 1) * grid_size_h, image.shape[0])
                    y_start = j * grid_size_w
                    y_end = min((j + 1) * grid_size_w, image.shape[1])
                    if np.any(inverted_mask[x_start:x_end, y_start:y_end]):  # 反転マスクがTrueの部分だけ処理
                        image[x_start:x_end, y_start:y_end] = 0
        masked_image = image
    
    elif option == 'all_checkerboard':
        # 偶数行と偶数列のピクセルを黒く塗りつぶす
        # image[::2, ::2] = 0
        # 奇数業と奇数列
        image[1::2, 1::2] = 0

        masked_image = image
    
    else:  # Simple binary mask
        masked_image = np.where(rgb_mask, image, 0)
    return masked_image


def show_masked_image(image: np.ndarray, mask: np.ndarray, ax: Axes) -> None:
    masked_image = apply_mask(image, mask)
    ax.imshow(masked_image)


def overlay_masks(image: np.ndarray, masks: List[np.ndarray], apply_effect: bool = False, effect_option: str = 'blur', alpha: float = 0.8, blur_strength: int = 10, block_size: int = 10) -> np.ndarray:
    combined_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined_mask |= mask
    if apply_effect:
        overlay_image = apply_mask(image, combined_mask, option=effect_option, alpha=alpha, blur_strength=blur_strength, block_size=block_size)
    else:
        rgb_mask = np.stack([combined_mask] * 3, axis=-1)
        overlay_image = np.zeros_like(image)
        overlay_image[rgb_mask] = image[rgb_mask]
    return overlay_image


def marking_keypoints(image: np.ndarray, key_points: List[Tuple[int, int]]) -> np.ndarray:
    for (x, y) in key_points:
        cv2.circle(image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    return image


def select_largest_mask(masks):
    mask_sizes = masks.sum(axis=(1, 2))
    largest_mask_index = mask_sizes.argmax()
    return masks[largest_mask_index]

