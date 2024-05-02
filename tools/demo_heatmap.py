import mmcv
from mmcv import imread, imwrite
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

import sys
sys.path.append('/home/moriki/PoseEstimation/mmpose')

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Specify the image file
img_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-origin/100000.jpg'

# Configuration files and model checkpoints
pose_config = 'configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py'
pose_checkpoint = 'models/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth'
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

device = 'cuda:0'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

# Initialize the detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)

# Initialize the pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# Load the image
image = imread(img_path)

# Run detection
person_bboxes = inference_detector(detector, image)
# Filter out low confidence detections
person_bboxes = [bbox for bbox in person_bboxes[0] if bbox[4] > 0.3]

# Run pose estimation
pose_results = inference_topdown(
    pose_estimator,
    image,
    person_bboxes,
    bbox_thr=0.3,
    format='xyxy'
)

# Initialize the visualizer with dataset meta
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

# Visualize results
vis_image = visualizer.show_result(
    image,
    pose_results,
    show=False
)

# Save visualization
output_path = 'output_pose.jpg'
imwrite(vis_image, output_path)

print(f'Visualization saved to {output_path}')
