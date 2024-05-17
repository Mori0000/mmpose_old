import json
import numpy as np
import cv2
import os

# 各関節間の接続情報
skeleton = [
    [12, 13],
    [13, 0],
    [13, 1],
    [0, 2],
    [2, 4],
    [1, 3],
    [3, 5],
    [13, 7],
    [13, 6],
    [7, 9],
    [9, 11],
    [6, 8],
    [8, 10]
]

def process_image(img_id, json_path, images_path, output_path, min_size=20):
    try:
        # JSONファイルを読み込む
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # 該当するimage_idのエントリを抽出
        annotations = [entry for entry in json_data['annotations'] if entry['image_id'] == img_id]
        if not annotations:
            print(f'No annotations found for image_id: {img_id}')
            return

        # 画像の読み込み
        image_path = os.path.join(images_path, f'{img_id}.jpg')  # 画像のパスは環境に合わせて調整 
        if not os.path.exists(image_path):
            print(f'Image file not found: {image_path}')
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f'Failed to read image: {image_path}')
            return

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path+'/Kpt', exist_ok=True)
        os.makedirs(output_path+'/img', exist_ok=True)

        image_height, image_width = image.shape[:2]

        # オリジナル画像のコピーを作成
        image_with_keypoints = image.copy()

        # 各アノテーションに対して処理を行う
        for idx, annotation in enumerate(annotations):
            print(f'Processing annotation {idx} for image_id: {img_id}')
            
            # キーポイントを抽出
            keypoints = np.array(annotation['keypoints']).reshape(-1, 3)
            visible_keypoints = keypoints[(keypoints[:, 2] > 0) & ((keypoints[:, 0] != 0) | (keypoints[:, 1] != 0))]  # 可視キーポイントのみで、座標が(0,0)でないもの

            if visible_keypoints.size == 0:
                print(f'No valid keypoints for image_id: {img_id}, annotation index: {idx}')
                continue

            # キーポイントを画像に重畳表示
            for i, (x, y, v) in enumerate(keypoints):
                if (x != 0 or y != 0):
                    color = (0, 255, 0) if v > 0 else (0, 0, 255)
                    cv2.circle(image_with_keypoints, (int(x), int(y)), 5, color, -1)

            # 各関節を線でつなぐ
            for (start, end) in skeleton:
                if (keypoints[start][0] != 0 or keypoints[start][1] != 0) and (keypoints[end][0] != 0 or keypoints[end][1] != 0):
                    start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
                    end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
                    cv2.line(image_with_keypoints, start_point, end_point, (255, 0, 0), 2)

            # キーポイントの最大最小値を計算
            min_x, min_y = visible_keypoints[:, :2].min(axis=0)
            max_x, max_y = visible_keypoints[:, :2].max(axis=0)

            # 切り出し領域の計算（1.1倍）
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            width, height = (max_x - min_x), (max_y - min_y)
            min_x, min_y = center_x - width / 2, center_y - height / 2
            max_x, max_y = center_x + width / 2, center_y + height / 2

            # 画像サイズの上限と下限を超えないように調整
            min_x = max(0, int(min_x))
            min_y = max(0, int(min_y))
            max_x = min(image_width, int(max_x))
            max_y = min(image_height, int(max_y))

            # 切り出し領域の計算
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

            # 小さすぎる領域を排除
            x, y, w, h = bbox
            if w <= min_size or h <= min_size:
                print(f'Region too small for image_id: {img_id}, annotation index: {idx}')
                continue

            # 画像の切り出し（オリジナル）
            cropped_image = image[y:y+h, x:x+w]

            # 画像の切り出し（キーポイント重畳）
            cropped_image_with_keypoints = image_with_keypoints[y:y+h, x:x+w]

            # 赤い楕円形の枠を描く
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(cropped_image_with_keypoints, center, axes, 0, 0, 360, (0, 0, 255), 2)

            # 切り出した画像を保存
            output_image_path = os.path.join(output_path, f'img/{img_id}_cropped_{idx}.jpg')
            cv2.imwrite(output_image_path, cropped_image)
            print(f'Cropped image saved: {output_image_path}')

            # 切り出した画像（キーポイント重畳）を保存
            output_image_with_keypoints_path = os.path.join(output_path, f'Kpt/{img_id}_cropped_with_keypoints_{idx}.jpg')
            cv2.imwrite(output_image_with_keypoints_path, cropped_image_with_keypoints)
            print(f'Cropped image with keypoints saved: {output_image_with_keypoints_path}')

        # オリジナル画像にキーポイントを重畳表示した画像を保存
        output_full_image_with_keypoints_path = os.path.join(output_path, f'Kpt/{img_id}_full_with_keypoints.jpg')
        cv2.imwrite(output_full_image_with_keypoints_path, image_with_keypoints)
        print(f'Full image with keypoints saved: {output_full_image_with_keypoints_path}')

    except FileNotFoundError:
        print(f'Warning: File {json_path} not found. Skipping.')
    except Exception as e:
        print(f'Error processing image_id {img_id}: {e}. Skipping.')

# パスの設定（環境に合わせて変更）
json_path = '/home/moriki/PoseEstimation/mmpose/data/crowdpose/annotations/mmpose_crowdpose_train.json'
images_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-origin'  # 画像が保存されているディレクトリ
output_path = '/home/moriki/PoseEstimation/mmpose/data/outputs/cropped-images/gts4'  # 切り出した画像を保存するディレクトリ

# 処理する画像IDの範囲
for img_id in range(100000, 100010):
    process_image(img_id, json_path, images_path, output_path)
