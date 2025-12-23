import warnings
# 屏蔽apex提示
warnings.filterwarnings('ignore', message='Better speed can be achieved with apex installed')

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from util.misc import collate_fn_with_mask as collate_fn
from engine import evaluate
from models import build_model

from datasets import build_dataset, test_transforms

from util.logger import get_logger
from util.config import Config


def get_args_parser():
    parser = argparse.ArgumentParser('TransCP: Evaluation only', add_help=False)
    # 基础参数
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for testing')
    parser.add_argument('--seed', default=3407, type=int)

    # 配置文件中的核心参数
    parser.add_argument('--dataset', default='pathology2', type=str)
    parser.add_argument('--output_dir', default='outputs/pathology_reason_test/public', type=str)
    parser.add_argument('--checkpoint_best', action='store_true')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--freeze_epochs', default=10, type=int)
    parser.add_argument('--freeze_modules', nargs='+', default=['backbone'], type=str)
    parser.add_argument('--load_weights_path', default='pretrained_checkpoints/detr-r50.pth', type=str)
    parser.add_argument('--model_config', default=None, type=eval)  # 解析字典配置

    # 【新增】模型构建依赖的优化器/学习率参数（必须添加）
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # 解决AttributeError的核心
    parser.add_argument('--lr_vis_enc', default=1e-5, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--freeze_param_names', type=list, default=[])
    parser.add_argument('--freeze_losses', type=list, default=[])

    # Model parameters
    parser.add_argument('--resume', default='/home/sse3090/jiangjiwei/TransCP/outputs/pathology_reason/public/checkpoint_best_acc.pth', 
                        help='resume from checkpoint (evaluation model path)')
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_path', default='pretrained_checkpoints/resnet50-19c8e357.pth', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))

    # * Transformer (visual encoder)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=1, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # * Bert (language encoder)
    parser.add_argument('--bert_model', default='pretrained_checkpoints/bert-base-uncased.tar.gz', type=str)
    parser.add_argument('--bert_token_mode', default='pretrained_checkpoints/bert_base_uncased', type=str)
    parser.add_argument('--bert_output_dim', default=768, type=int)
    parser.add_argument('--bert_output_layers', default=12, type=int)
    parser.add_argument('--max_query_len', default=40, type=int)
    parser.add_argument('--bert_enc_num', default=12, type=int)

    # Loss (评估时仍需criterion计算损失指标)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--loss_loc', default='loss_boxes', type=str)
    parser.add_argument('--box_xyxy', action='store_true')
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    # dataset parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--split_root', default='/home/sse3090/jiangjiwei/TransCP/split/data')
    parser.add_argument('--test_split', default='testA')
    parser.add_argument('--img_size', default=640)
    parser.add_argument('--cache_images', action='store_true')
    parser.add_argument('--save_pred_path', default='predictions.json', help='path to save prediction results')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=True, type=boolean_string)
    parser.add_argument('--batch_size_test', default=1, type=int)  # 评估批次大小
    parser.add_argument('--test_transforms', default=test_transforms)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # configure file
    parser.add_argument('--config', default='configs/TransCP_R50_pathology2.py', type=str)
    
    return parser


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main(args):
    # 初始化分布式模式
    utils.init_distributed_mode(args)

    # 创建日志器
    eval_output_dir = Path(args.output_dir) / "eval_results"  # 评估结果单独存放
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("evaluation", eval_output_dir, utils.get_rank(), filename='eval.log')
    logger.info("===== TransCP Evaluation Mode =====")
    logger.info(f"Loaded config from: {args.config}")
    logger.info(args)

    # 设置设备
    device = torch.device(args.device)
    if not torch.cuda.is_available() and args.device == 'cuda':
        logger.warning("CUDA is not available, switching to CPU")
        device = torch.device('cpu')

    # 固定随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型、损失函数和后处理器（传入model_config）
    model, criterion, postprocessor = build_model(args)
    model.to(device)
    logger.info(f"Model built with config: {args.model_config}")

    # 分布式包装
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 加载预训练/权重文件
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            logger.info(f"Loaded model weights from checkpoint: {args.resume}")
        else:
            model_without_ddp.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded model weights (direct state dict) from: {args.resume}")
    elif args.load_weights_path:
        model_without_ddp.load_pretrained_weights(args.load_weights_path)
        logger.info(f"Loaded pretrained weights from: {args.load_weights_path}")
    else:
        logger.warning("No model weights loaded! Using random initialization.")

    # 构建测试数据集
    dataset_test = build_dataset(test=True, args=args)
    logger.info(f"Test dataset ({args.dataset}) size: {len(dataset_test)}")

    # 构建数据加载器
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(
        dataset_test, args.batch_size_test, sampler=sampler_test,
        pin_memory=args.pin_memory, drop_last=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )

    # 执行评估
    logger.info("Starting evaluation...")
    start_time = time.time()
    test_stats, test_acc, test_time = evaluate(
        model, criterion, postprocessor, data_loader_test, device, str(eval_output_dir / args.save_pred_path)
    )
    eval_time = time.time() - start_time

    # 打印并记录评估结果
    logger.info("===== Evaluation Results =====")
    # 损失指标
    logger.info('Loss Stats | ' + ' | '.join([f'{k}: {v:.4f}' for k, v in test_stats.items()]))
    # 精度指标
    logger.info('Accuracy   | ' + ' | '.join([f'{k}: {v:.4f}' for k, v in test_acc.items()]))
    # 时间统计
    logger.info(f"Total evaluation time: {datetime.timedelta(seconds=int(eval_time))}")
    logger.info(f"Test time details: {test_time}")

    # 保存评估结果到文件
    if utils.is_main_process():
        results = {
            'dataset': args.dataset,
            'test_split': args.test_split,
            'model_checkpoint': args.resume,
            'test_stats': test_stats,
            'test_acc': test_acc,
            'test_time': test_time,
            'total_eval_time': str(datetime.timedelta(seconds=int(eval_time)))
        }
        with open(eval_output_dir / 'eval_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to: {eval_output_dir / 'eval_results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransCP Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    # 加载配置文件
    if args.config:
        cfg = Config(args.config)
        cfg.merge_to_args(args)
    # 执行评估
    main(args)