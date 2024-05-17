import torch
import clip
from PIL import Image, ImageOps, ImageDraw
import json
import numpy as np
import cv2
import os

def plot_keypoints(img, keypoints, scores):
    """Plot keypoints on the image using cv2 with color based on confidence scores."""
    img = np.array(img)

    # Draw connections with a default color (e.g., white)
    connections = [
        (12, 13), (13, 0), (13, 1), (0, 2), (1, 3), (2, 4), (3, 5),
        (0, 6), (1, 7), (6, 8), (7, 9), (8, 10), (9, 11)
    ]
    for start, end in connections:
        start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
        end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
        cv2.line(img, start_point, end_point, (255, 255, 255), 2)  # White lines for visibility

    # Draw keypoints with colors based on scores
    for idx, (x, y) in enumerate(keypoints):
        score = scores[idx]
        color = (255 * (1 - score), 0, 255 * score)  # Interpolating between blue (low score) and red (high score)
        cv2.circle(img, (int(x), int(y)), 5, color, -1)

    return Image.fromarray(img)

def resize_with_padding(img, target_size=(224, 224)):
    """Resize the image while keeping the aspect ratio and adding padding."""
    original_width, original_height = img.size
    target_width, target_height = target_size

    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded_img = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color='black')

    return padded_img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    json_path = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data.json'
    with open(json_path, 'r') as f:
        Kpt_data = json.load(f)

    output_dir = '/home/moriki/PoseEstimation/mmpose/outputs/check_clip_circle3'
    os.makedirs(output_dir, exist_ok=True)
    preds = []
    for img_id in [100003, 100007, 100010]:
        pred = []
        img_path = f'/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-origin/{img_id}.jpg'
        keypoints_set, scores_set = None, None
        for data in Kpt_data:
            if data['img_id'] == img_id:
                keypoints_set = data['pred_keypoints']
                scores_set = data['keypoint_scores']
                break

        if not keypoints_set or not scores_set:
            print(f"No keypoints or scores found for img_id: {img_id}. Skipping.")
            continue

        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print(f"Image file not found for img_id: {img_id}. Skipping.")
            continue

        for i, (keypoints, scores) in enumerate(zip(keypoints_set, scores_set)):
            keypoints_array = np.array(keypoints)
            scores_array = np.array(scores)

            min_x = int(max(0, np.min(keypoints_array[:, 0]) * 0.9))
            max_x = int(max(min_x + 1, np.max(keypoints_array[:, 0]) * 1.1))
            min_y = int(max(0, np.min(keypoints_array[:, 1]) * 0.9))
            max_y = int(max(min_y + 1, np.max(keypoints_array[:, 1]) * 1.1))

            crop_img = img.crop((min_x, min_y, max_x, max_y))

            # Convert PIL image to OpenCV format
            crop_img_cv = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
            w, h = crop_img_cv.shape[1], crop_img_cv.shape[0]

            # Calculate the thickness of the ellipse line based on the size of the cropped image
            line_thickness = max(1, int(0.05 * min(w, h)))  # Adjust the factor as needed

            # Draw a red ellipse
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(crop_img_cv, center, axes, 0, 0, 360, (0, 0, 255), line_thickness)

            # Convert back to PIL format
            crop_img = Image.fromarray(cv2.cvtColor(crop_img_cv, cv2.COLOR_BGR2RGB))

            crop_img = resize_with_padding(crop_img, target_size=(224, 224))
            clip_image = preprocess(crop_img).unsqueeze(0).to(device)
            text = clip.tokenize(["A image of a human", "A image of an object"]).to(device)

            with torch.no_grad():
                logits_per_image, logits_per_text = model(clip_image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            prob_formatted = f"{probs[0][0]:.2f}_{probs[0][1]:.2f}"
            plot_img = plot_keypoints(img, keypoints_array, scores_array)
            crop_plot_img = plot_img.crop((min_x, min_y, max_x, max_y))
            crop_img.save(f'{output_dir}/{img_id}_{i}.jpg')
            crop_plot_img.save(f'{output_dir}/{img_id}_{i}_{prob_formatted}.jpg')
            # print("Label probabilities:", probs)
            if probs[0][0] > 0.5:
                pred.append(0)
            else:
                pred.append(1)        
        preds.append(pred)
    # score
    ans = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    ]
    
    for i, (pred, an) in enumerate(zip(preds, ans)):
        print(f'img_id: {i}')
        print(f'pred: {pred}')
        print(f'ans: {an}')
        score = np.sum(np.array(pred) == np.array(an)) / len(pred) * 100
        print(f'score: {score:.2f}%')
        print('------------------------------------------------------------------')

if __name__ == "__main__":
    main()
