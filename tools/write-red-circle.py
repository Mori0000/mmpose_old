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

def process_image(img_id, json_path, images_path, output_path):
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
output_path = '/home/moriki/PoseEstimation/mmpose/data/outputs/cropped-images/gts5'  # 切り出した画像を保存するディレクトリ

# 処理する画像IDの範囲
for img_id in range(100000, 100010):
    process_image(img_id, json_path, images_path, output_path)
