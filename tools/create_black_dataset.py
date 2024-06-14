import os
import shutil

# ディレクトリのパス
images_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images'
black_img_root_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-black'

# 黒背景画像のディレクトリ内のディレクトリリストを取得
black_img_dirs = [d for d in os.listdir(black_img_root_path) if os.path.isdir(os.path.join(black_img_root_path, d))]

# 共通の画像ファイルをコピー
copied_files_count = 0
for dir_name in black_img_dirs:
    img_id = dir_name
    src_file = os.path.join(black_img_root_path, dir_name, 'overlayed_image.jpg')
    dst_file = os.path.join(images_path, f'{img_id}.jpg')
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        copied_files_count += 1

print(f'Copied {copied_files_count} files from {black_img_root_path} to {images_path}.')
