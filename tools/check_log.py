import json

# ログファイルのパス
log_file_path = '/home/moriki/PoseEstimation/mmpose/example_20240508_0007.log'

# データ構造を初期化
logged_data = {'outputs': []}

# データを格納するリスト
data_list = []

with open(log_file_path, 'r') as file:
    for line in file:
        try:
            # 一行を JSON として解析
            data = json.loads(line.strip())  # strip() を使用して余分な空白や改行を削除
            data_list.append(data)
        except json.JSONDecodeError as e:
            # JSON 解析エラーが発生した場合は、エラーメッセージを出力
            print(f"Error parsing JSON: {e} in line: {line}")

for sample in logged_data['outputs']:
    image_id = sample.img_id
    gt_keypoints = sample.gt_instances.keypoints.tolist()
    pred_keypoints = sample.pred_instances.keypoints.tolist()
    
    print(f"Image ID: {image_id}")
    print("Ground Truth Keypoints:")
    print(gt_keypoints)
    print("Predicted Keypoints:")
    print(pred_keypoints)

    # If you want to save this to a JSON file:
    import json
    data_to_save = {
        "image_id": image_id,
        "ground_truth_keypoints": gt_keypoints,
        "predicted_keypoints": pred_keypoints
    }
    
    with open('keypoints_data.json', 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    print("Data has been saved to 'keypoints_data.json'")

