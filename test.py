r""" Cross-Domain Few-Shot Semantic Segmentation testing code """
import argparse

import torch.nn as nn
import torch

from model.patnet import PATNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def test(model, dataloader, nshot):
    r""" Test PATNet """

    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. PATNetworks forward pass
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        assert pred_mask.size() == batch['query_mask'].size()
        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Cross-Domain Few-Shot Semantic Segmentation Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../CDFSL')
    parser.add_argument('--benchmark', type=str, default='fss', choices=['fss', 'deepglobe', 'isic', 'lung'])
    parser.add_argument('--logpath', type=str, default='./')
    parser.add_argument('--bsz', type=int, default=30)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='path_to_your_trained_model')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = PATNetwork(args.backbone)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test PATNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
