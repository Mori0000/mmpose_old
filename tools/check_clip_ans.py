import torch
import clip
from PIL import Image, ImageOps
import json
import numpy as np
import cv2

'''
入力された画像のidに対し、その画像に対する予測されたキーポイントとスコアを取得し、
その画像をクリップモデルに入力し、その画像に対するラベルの確率を取得。

画像を切り抜く部分は、キーポイント12と13のスコアが0.9以上の場合は顔を切り抜き、
それ以外の場合は全身を切り抜く（キーポイント全体の最大最小値から1.1倍の領域を切り出す）。
'''


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
    # 元の画像サイズとターゲットサイズ
    original_width, original_height = img.size
    target_width, target_height = target_size

    # アスペクト比を計算
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # アスペクト比を保持してリサイズ
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # パディングを追加して中央に配置
    padded_img = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color='black')

    return padded_img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    json_path = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data.json'
    with open(json_path, 'r') as f:
        Kpt_data = json.load(f)

    for img_id in range(100000, 100040):
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

            # Check confidence for keypoints 12 and 13
            if scores_array[12] >= 0.9 and scores_array[13] >= 0.9:
                # Calculate crop for face
                x1, y1 = keypoints_array[12]
                x2, y2 = keypoints_array[13]
                distance = int(((x2 - x1)**2 + (y2 - y1)**2)**0.5)
                min_x = max(0, int((x1 + x2) / 2 - distance / 2))
                max_x = min(img.width, int((x1 + x2) / 2 + distance / 2))
                min_y = max(0, int((y1 + y2) / 2 - distance / 2))
                max_y = min(img.height, int((y1 + y2) / 2 + distance / 2))
            else:
                # Calculate crop for full body
                min_x = int(max(0, np.min(keypoints_array[:, 0])*0.9))
                max_x = int(max(min_x + 1, np.max(keypoints_array[:, 0])*1.1))
                min_y = int(max(0, np.min(keypoints_array[:, 1])*0.9))
                max_y = int(max(min_y + 1, np.max(keypoints_array[:, 1])*1.1))

            crop_img = img.crop((min_x, min_y, max_x, max_y))
            clip_image = preprocess(crop_img).unsqueeze(0).to(device)
            text = clip.tokenize(["A image of a human", "A image of an object"]).to(device)

            with torch.no_grad():
                logits_per_image, logits_per_text = model(clip_image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            prob_formatted = f"{probs[0][0]:.2f}_{probs[0][1]:.2f}"
            plot_img = plot_keypoints(img, keypoints_array, scores)  # Pass scores array to plot_keypoints
            crop_plot_img = plot_img.crop((min_x, min_y, max_x, max_y))
            crop_img.save(f'/home/moriki/PoseEstimation/mmpose/outputs/check_clip2/{img_id}_{i}.jpg')
            crop_plot_img.save(f'/home/moriki/PoseEstimation/mmpose/outputs/check_clip2/{img_id}_{i}_{prob_formatted}.jpg')
            print("Label probabilities:", probs)

# Call the main function
if __name__ == "__main__":
    main()