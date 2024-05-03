import mmcv
from mmcv import imread
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

from IPython.display import Image, display
import tempfile
import os.path as osp
import cv2

from mmdet.apis import inference_detector, init_detector
has_mmdet = True

def visualize_img(img_path, detector, pose_estimator, visualizer,
                  show_interval, out_file):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    img = mmcv.imread(img_path, channel_order='rgb')

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=True,
        draw_bbox=False,  # バウンディングボックスは不要な場合はFalseに
        show=False,
        wait_time=show_interval,
        out_file=out_file
        )


def main():
    try:
        from mmdet.apis import inference_detector, init_detector
        has_mmdet = True
    except (ImportError, ModuleNotFoundError):
        has_mmdet = False

    img = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-origin/100000.jpg'

    # Correct the path to the pose estimation model configuration file
    pose_config = '/home/moriki/PoseEstimation/mmpose/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py'
    pose_checkpoint = '/home/moriki/PoseEstimation/mmpose/models/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth'
    det_config = '/home/moriki/PoseEstimation/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    device = 'cuda:0'
    cfg_options = None  # Assuming default configuration should work unless you have specific modifications to make

    # Initialize the object detector
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

    # Initialize the visualizer with specific settings
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 1
    print('-------config-------')
    print(pose_estimator.cfg.visualizer)
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # Dataset meta-information is loaded from the checkpoint and passed to the model
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    visualize_img(
        img,
        detector,
        pose_estimator,
        visualizer,
        show_interval=0,
        out_file='/home/moriki/PoseEstimation/mmpose/outputs/heatmap-img/pose_heatmap_result.jpg')

    vis_result = visualizer.get_image()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = osp.join(tmpdir, 'pose_results.png')
        cv2.imwrite(file_name, vis_result[:,:,::-1])
        display(Image(file_name))

if __name__ == '__main__':
    main()