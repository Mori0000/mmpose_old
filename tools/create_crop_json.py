import json

'''
Cropした画像から切り取った座標をJSON形式に変換するスクリプト
これで生成したjsonファイルを評価関数で読み込むことで、切り取った座標分を補正して評価を行う
'''


# テキストファイルのパスを変数に保存
txt_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-cropped/results.txt'

# テキストファイルの内容を読み込み
data = []
with open(txt_path, 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) != 5:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        try:
            img_id = int(parts[0].strip().split('/')[-1].split('.')[0])
            x1 = int(parts[1].strip(' ()'))
            y1 = int(parts[2].strip(' ()'))
            x2 = int(parts[3].strip(' ()'))
            y2 = int(parts[4].strip(' ()'))
            data.append({
                "img_id": img_id,
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })
        except ValueError as e:
            print(f"Skipping line due to error: {line.strip()} - {e}")
            continue

# JSON形式に変換
json_data = json.dumps(data, indent=4, ensure_ascii=False)

# JSONファイルに保存
json_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-cropped/results.json'
with open(json_path, 'w') as crop_json_file:
    json.dump(data, crop_json_file, indent=4, ensure_ascii=False)
