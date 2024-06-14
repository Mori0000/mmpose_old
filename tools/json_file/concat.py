import json

'''
jsonファイルを結合する
'''
from datetime import datetime

def concat_json_files(json_files, output_file):
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data += json.load(f)
    with open(output_file, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    today_date = datetime.now().strftime("%Y%m%d")

    json_0 = f'/home/moriki/PoseEstimation/mmpose/tools/json_file/origin/mmpose_data_{today_date}_0.json'
    json_1 = f'/home/moriki/PoseEstimation/mmpose/tools/json_file/origin/mmpose_data_{today_date}_1.json'
    json_2 = f'/home/moriki/PoseEstimation/mmpose/tools/json_file/origin/mmpose_data_{today_date}_2.json'
    json_3 = f'/home/moriki/PoseEstimation/mmpose/tools/json_file/origin/mmpose_data_{today_date}_3.json'
    
    json_files = [json_0, json_1, json_2, json_3]
    
    output_file = f'/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data_{today_date}.json'
    
    concat_json_files(json_files, output_file)
    print(f"データを {output_file} に保存しました")


'''
読み込んだjsonファイルのデータの概要を表示する
'''
# import json

# # JSONファイルを読み込む
# file_path = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data.json'
# with open(file_path, 'r') as file:
#     data = json.load(file)

# # データの概要を表示
# print("データの長さ:", len(data))
# if data:
#     print("最初の要素のキー:", data[0].keys())

# # 全てのデータのキーを確認する（もし異なる形状のデータがある場合）
# all_keys = set()
# for item in data:
#     all_keys.update(item.keys())
# print("全てのアイテムで使われるキー:", all_keys)