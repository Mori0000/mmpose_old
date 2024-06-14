import os
import shutil

# ディレクトリのパス
images_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images'
cropped_images_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-cropped'

# imagesディレクトリ内の画像ファイルのリスト
images_files = set(os.listdir(images_path))

# images_croppedディレクトリ内の画像ファイルのリスト
cropped_images_files = set(os.listdir(cropped_images_path))

# 両方のディレクトリに存在する画像ファイルのリスト
common_files = images_files.intersection(cropped_images_files)

# 共通の画像ファイルをimages_croppedからimagesにコピー
for file_name in common_files:
    src_file = os.path.join(cropped_images_path, file_name)
    dst_file = os.path.join(images_path, file_name)
    shutil.copy2(src_file, dst_file)

print(f'Copied {len(common_files)} files from {cropped_images_path} to {images_path}.')
