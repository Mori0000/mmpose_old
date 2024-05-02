import json
import cv2
import numpy as np

# 確認したい画像の番号を選択
img_num = 100007
img_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-origin'
data_path = '/home/moriki/PoseEstimation/mmpose/outputs/inferencer-demo'

# JSONファイルからデータを読み込む
with open(f'{data_path}/{img_num}.json') as f:
    data = json.load(f)

image = cv2.imread(f'{img_path}/{img_num}.jpg')  # 画像ファイルのパスを指定してください

# 複数のキーポイントを処理する
for idx, detection in enumerate(data):
    # 新しい画像を作成して元の画像をコピーする（重ねて描画しないようにするため）
    image_to_draw = image.copy()

    # キーポイントとスコアを抽出
    keypoints = np.array(detection["keypoints"])
    keypoint_scores = np.array(detection["keypoint_scores"])

    # BBoxを抽出
    bbox = np.array(detection["bbox"][0])

    # 画像にキーポイントを描画
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        score = keypoint_scores[i]
        if score > 0.01:  # スコアが0.5より大きい場合のみ描画
            cv2.circle(image_to_draw, (int(x), int(y)), 5, (0, 255, 0), -1)  # キーポイントを緑の円で描画
            # キーポイントのスコアを画像に描画
            cv2.putText(image_to_draw, f'{score:.2f}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 画像にBBoxを描画
    cv2.rectangle(image_to_draw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)  # BBoxを赤い枠で描画
    # BBoxのスコアを画像名に追加
    save_name_with_score = f'{data_path}/{img_num}_{idx}_bbox{detection["bbox_score"]:.2f}.jpg'

    # 画像を保存
    cv2.imwrite(save_name_with_score, image_to_draw)
    print(f'保存された画像: {save_name_with_score}')

