import json

'''
jsonファイルを結合する
'''

def concat_json_files(json_files, output_file):
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data += json.load(f)
    with open(output_file, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    json_0 = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data_0.json'
    json_1 = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data_1.json'
    json_2 = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data_2.json'
    json_3 = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data_3.json'
    
    json_files = [json_0, json_1, json_2, json_3]
    output_file = '/home/moriki/PoseEstimation/mmpose/tools/json_file/mmpose_data.json'
    concat_json_files(json_files, output_file)


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