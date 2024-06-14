import torch, torchvision
import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import os
import cv2

# my library
import sys
sys.path.append('/home/moriki/PoseEstimation/mmpose')

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector

current_pythonpath = os.environ.get('PYTHONPATH', '')
new_path = '/home/moriki/PoseEstimation/mmpose/'
if new_path not in current_pythonpath:
    os.environ['PYTHONPATH'] = f"{current_pythonpath}:{new_path}" if current_pythonpath else new_path

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
        draw_bbox=True,
        show=False,
        wait_time=show_interval,
        out_file=out_file,
        kpt_thr=0.3)

    visualize_img(
        img,
        detector,
        pose_estimator,
        visualizer,
        show_interval=0,
        out_file=None)

    vis_result = visualizer.get_image()

    from IPython.display import Image, display
    import tempfile
    import os.path as osp
    import cv2
    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = osp.join(tmpdir, 'pose_results.png')
        cv2.imwrite(file_name, vis_result[:,:,::-1])
        display(Image(file_name))
        cv2_imshow(vis_result[:,:,::-1]) #RGB2BGR to fit cv2

    return vis_result

def main():
    # Load model
    pose2d_cfg = 'configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py'
    pose2d_weights = 'models/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth'
    # pose3d_cfg = 'configs/body/3d_kpt_sview_rgb_img/topdown_heatmap/coco/resnet_50_coco_256x192.py'
    # pose3d_weights = 'checkpoints/resnet_50_coco_256x192-ec54d7f3_20200708.pth'
    det_cfg = 'configs/body/yoloxpose/yoloxpose_sview_rgb_img/yoloxpose_sview_rgb_img_256x192.py'
    det_weights = 'checkpoints/yoloxpose_sview_rgb_img_256x192-ec54d7f3_20200708.pth'

    pose2d = init_pose_estimator(pose2d_cfg, pose2d_weights, device='cuda:0')
    # pose3d = init_pose_estimator(pose3d_cfg, pose3d_weights, device='cuda:0')
    detector = init_detector(det_cfg, det_weights, device='cuda:0')
    img_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images-origin/100000.jpg'
    visualizer = VISUALIZERS.get('topdown_heatmap')
    out_file = '/home/moriki/PoseEstimation/mmpose/outputs/heatmap-img/pose_estimation_result.jpg'
    visualizer = VISUALIZERS.get('topdown_heatmap')
    visualize_img(img_path, detector, pose2d, visualizer, show_interval=0, out_file=out_file)

if __name__ == '__main__':
    main()