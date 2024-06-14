# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner

import pickle
import os
import copy
import json
import logging
import torch

import datetime
import numpy as np


'''

'''

def parse_args():                                                               # ここで引数をパースしている
    parser = argparse.ArgumentParser(                                           # argparse.ArgumentParser()でパーサーを作成
        description='MMPose test (and eval) model')                             # パーサーの説明を追加
    parser.add_argument('config', help='test config file path')                 # config引数を追加
    parser.add_argument('checkpoint', help='checkpoint file')                   # ckeckpoint引数を追加
    parser.add_argument(                                                        
        '--work-dir', help='the directory to save evaluation results')          # work-dir引数を追加 
    parser.add_argument('--out', help='the file to save metric results.')       # --out引数を追加
    parser.add_argument(
        '--dump',                                                               # --dump引数を追加
        type=str,                                                               # 引数の型を指定
        help='dump predictions to a pickle file for offline evaluation')        # ヘルプメッセージを追加
    parser.add_argument(
        '--cfg-options',                                                        # --cfg-options引数を追加
        nargs='+',                                                              # 引数の数を指定
        action=DictAction,                                                      # アクションを指定
        default={},                                                             # デフォルト値を指定
        help='override some settings in the used config, the key-value pair '   # ヘルプメッセージを追加
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--show-dir',
        # default='/home/moriki/PoseEstimation/mmpose/data/outputs/mmpose/show',    # ここのコメントアウト外せば、結果が保存される
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        # default='/home/moriki/PoseEstimation/mmpose/data/outputs/mmpose/shows',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--badcase',
        action='store_true',
        default=False,
        help='whether analyze badcase in test')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):                              # ここで引数をマージしている
    """Merge CLI arguments to config."""

    cfg.launcher = args.launcher                        # cfg.launcherにargs.launcherを代入
    cfg.load_from = args.checkpoint                     # cfg.load_fromにargs.checkpointを代入

    # -------------------- work directory --------------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:                                                   # args.work_dirがNoneでない場合
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir                                                # cfg.work_dirにargs.work_dirを代入
    elif cfg.get('work_dir', None) is None:                                         # cfg.get('work_dir', None)がNoneの場合
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',                                      # cfg.work_dirに'./work_dirs'とargs.configのベースネームを結合したものを代入
                                osp.splitext(osp.basename(args.config))[0])         # osp.splitext(osp.basename(args.config))[0]はargs.configのベースネームを取得している

    # -------------------- visualization --------------------
    if (args.show and not args.badcase) or (args.show_dir is not None):             # ここでargs.show_dirがNoneでない場合に、cfg.default_hooks.visualization.out_dirにargs.show_dirを代入している
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True                               # cfg.default_hooks.visualization.enableにTrueを代入
        cfg.default_hooks.visualization.show = False \
            if args.badcase else args.show                                          # cfg.default_hooks.visualization.showにFalseを代入
        if args.show:                                                               # args.showがTrueの場合
            cfg.default_hooks.visualization.wait_time = args.wait_time      # cfg.default_hooks.visualization.wait_timeにargs.wait_timeを代入
        cfg.default_hooks.visualization.out_dir = args.show_dir             # cfg.default_hooks.visualization.out_dirにargs.show_dirを代入
        cfg.default_hooks.visualization.interval = args.interval            # cfg.default_hooks.visualization.intervalにargs.intervalを代入

    # -------------------- badcase analyze --------------------
    if args.badcase:                                                        # args.badcaseがTrueの場合
        assert 'badcase' in cfg.default_hooks, \
            'BadcaseAnalyzeHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`badcase=dict(type="BadcaseAnalyzeHook")`'

        cfg.default_hooks.badcase.enable = True                             # cfg.default_hooks.badcase.enableにTrueを代入
        cfg.default_hooks.badcase.show = args.show                          # cfg.default_hooks.badcase.showにargs.showを代入
        if args.show:                                                       # args.showがTrueの場合
            cfg.default_hooks.badcase.wait_time = args.wait_time            # cfg.default_hooks.badcase.wait_timeにargs.wait_timeを代入
        cfg.default_hooks.badcase.interval = args.interval                  # cfg.default_hooks.badcase.intervalにargs.intervalを代入

        metric_type = cfg.default_hooks.badcase.get('metric_type', 'loss')  # cfg.default_hooks.badcase.get('metric_type', 'loss')をmetric_typeに代入
        if metric_type not in ['loss', 'accuracy']:                         # metric_typeが['loss', 'accuracy']に含まれていない場合
            raise ValueError('Only support badcase metric type'             # ValueErrorを発生させる
                             "in ['loss', 'accuracy']")                     # エラーメッセージを追加

        if metric_type == 'loss':                                           # metric_typeが'loss'の場合
            if not cfg.default_hooks.badcase.get('metric'):                 # cfg.default_hooks.badcase.get('metric')がNoneの場合
                cfg.default_hooks.badcase.metric = cfg.model.head.loss      # cfg.default_hooks.badcase.metricにcfg.model.head.lossを代入
        else:                                                               # metric_typeが'loss'でない場合
            if not cfg.default_hooks.badcase.get('metric'):                 # cfg.default_hooks.badcase.get('metric')がNoneの場合
                cfg.default_hooks.badcase.metric = cfg.test_evaluator       # cfg.default_hooks.badcase.metricにcfg.test_evaluatorを代入

    # -------------------- Dump predictions --------------------
    if args.dump is not None:                                               # args.dumpがNoneでない場合
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)     # dump_metricに{'type': 'DumpResults', 'out_file_path': args.dump}を代入
        if isinstance(cfg.test_evaluator, (list, tuple)):                   # cfg.test_evaluatorがリストかタプルの場合
            cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]         # cfg.test_evaluatorにcfg.test_evaluatorとdump_metricを結合したものを代入
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]          # cfg.test_evaluatorに[cfg.test_evaluator, dump_metric]を代入

    # -------------------- Other arguments --------------------
    if args.cfg_options is not None:                                        # args.cfg_optionsがNoneでない場合
        cfg.merge_from_dict(args.cfg_options)                               # cfg.merge_from_dict(args.cfg_options)を実行

    return cfg


def convert_ndarray(data):
    """Recursively convert numpy arrays to lists."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_ndarray(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray(item) for item in data]
    return data

def append_to_json_file(file_path, new_data):
    """Append new data to a JSON file or create it if it does not exist."""
    if os.path.exists(file_path):
        with open(file_path, 'r+') as file:
            try:
                data = json.load(file)  # Existing data
                data.append(new_data)  # Append new data
                file.seek(0)  # Rewind to the start of the file
                json.dump(data, file, indent=4)
                file.truncate()  # Truncate file to new data size
            except json.JSONDecodeError:
                data = [new_data]  # Reset if file is corrupted
                file.seek(0)
                json.dump(data, file, indent=4)
    else:
        with open(file_path, 'w') as file:
            json.dump([new_data], file, indent=4)  # Start a new list with the data

class CustomRunner(Runner):
    @classmethod
    def from_cfg(cls, cfg) -> 'Runner':
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )
        return runner


    def test(self) -> dict:
        if self._test_loop is None:
            self._test_loop = self.build_test_loop(self._test_loop)  # ここはCustomRunnerで完全に制御
        self.call_hook('before_run')  # フックの呼び出しもCustomRunnerで制御
        self.load_or_resume()  # モデルロードの制御
        metrics = self.test_loop.run()  # この部分では基底クラスの実装を使う可能性がある
        self.call_hook('after_run')
        return metrics

    def call_hook(self, fn_name, **kwargs):
        today = datetime.datetime.now().strftime('%Y%m%d')
        json_path = f'/home/moriki/PoseEstimation/mmpose/tools/json_file/origin/mmpose_data_{today}_{torch.cuda.current_device()}.json'
        log_message = f'CustomRunner: {fn_name} called with '
        
        logged_data = {}
        
        if 'outputs' in kwargs:
            logged_data['outputs'] = kwargs['outputs']
            # crop_json_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/jsonfiles/kpt_all_gt.json'
            crop_json_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/jsonfiles/kpt_demo.json'
            
            with open(crop_json_path, 'r') as crop_json_file:
                crop_json_data = json.load(crop_json_file)
                
                if crop_json_data:
                    for data in crop_json_data:
                        if data['img_id'] == kwargs['outputs'][0].img_id:
                            x2 = data['coordinates']['x1']  # 小さい方を採用
                            y2 = data['coordinates']['y1']  # 小さい方を採用
                            kwargs['outputs'][0].pred_instances.keypoints += np.array([x2, y2])

        if logged_data:
            log_data = logged_data['outputs'][0]
            img_id = log_data.img_id
            pred_keypoints = log_data.pred_instances.keypoints
            keypoint_scores = log_data.pred_instances.keypoint_scores
            bboxes = log_data.pred_instances.bboxes
            # kwargs['outputs'][0].pred_instances.keypoints これが予測結果のキーポイント座標
            # Convert numpy arrays to lists
            pred_keypoints_list = convert_ndarray(pred_keypoints)
            keypoint_scores_list = convert_ndarray(keypoint_scores)
            bboxes_list = convert_ndarray(bboxes)

            # Prepare data to save
            data_to_save = {
                'img_id': img_id,
                'pred_keypoints': pred_keypoints_list,
                'keypoint_scores': keypoint_scores_list,
                'bboxes': bboxes_list,
            }

            # Append data to JSON file
            append_to_json_file(json_path, data_to_save)
        super().call_hook(fn_name, **kwargs)

        

def main():
    time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    logging.basicConfig(
        filename=f'example_{time}.log',  # ログを保存するファイル名に現在の日付と時間を反映
        filemode='a',            # 'a' は追記モード、'w' は上書きモード
        level=logging.DEBUG,     # ログレベル
        format='%(asctime)s - %(levelname)s - %(message)s'  # ログのフォーマット
    )
    # 残りのコードは同じです。
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    runner = CustomRunner.from_cfg(cfg)
    runner.test()
    # runner.after_test_epoch(metrics)

if __name__ == '__main__':
    main()
