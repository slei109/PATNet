r""" PATNetwork """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner


class PATNetwork(nn.Module):
    def __init__(self, backbone):
        super(PATNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.reference_layer3 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer3.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer3.bias, 0)
            self.reference_layer2 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer2.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer2.bias, 0)
            self.reference_layer1 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer1.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer1.bias, 0)
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.reference_layer3 = nn.Linear(2048, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer3.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer3.bias, 0)
            self.reference_layer2 = nn.Linear(1024, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer2.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer2.bias, 0)
            self.reference_layer1 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer1.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer1.bias, 0)
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats, support_mask.clone())
        query_feats, support_feats = self.Transformation_Feature(query_feats, support_feats, prototypes_f, prototypes_b)
        corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

        logit_mask = self.hpn_learner(corr)
        logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask

    def mask_feature(self, features, support_mask):
        eps = 1e-6
        prototypes_f = []
        prototypes_b = []
        bg_features = []
        mask_features = []
        for idx, feature in enumerate(features): ## [layernum, batchsize, C, H, W]
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            bg_mask = 1 - mask
            bg_features.append(features[idx] * bg_mask)
            mask_features.append(features[idx] * mask)
            features[idx] = features[idx] * mask
            ### prototype
            proto_f = features[idx].sum((2, 3))
            label_sum = mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            prototypes_f.append(proto_f)
            proto_b = bg_features[idx].sum((2, 3))
            label_sum = bg_mask.sum((2, 3))
            proto_b = proto_b / (label_sum + eps)
            prototypes_b.append(proto_b)
        return mask_features, prototypes_f, prototypes_b

    def Transformation_Feature(self, query_feats, support_feats, prototypes_f, prototypes_b):
        transformed_query_feats = []
        transformed_support_feats = []
        bsz = query_feats[0].shape[0]
        for idx, feature in enumerate(support_feats):
            C = torch.cat((prototypes_b[idx].unsqueeze(1), prototypes_f[idx].unsqueeze(1)), dim=1)
            eps = 1e-6
            if idx <= 3:
                R = self.reference_layer1.weight.expand(C.shape)
            elif idx <= 9:
                R = self.reference_layer2.weight.expand(C.shape)
            elif idx <= 12:
                R = self.reference_layer3.weight.expand(C.shape)
            power_R = ((R * R).sum(dim=2, keepdim=True)).sqrt()
            R = R / (power_R + eps)
            power_C = ((C * C).sum(dim=2, keepdim=True)).sqrt()
            C = C / (power_C + eps)

            P = torch.matmul(torch.pinverse(C), R)
            P = P.permute(0, 2, 1)

            init_size = query_feats[idx].shape
            query_feats[idx] = query_feats[idx].view(bsz, C.size(2), -1)
            transformed_query_feats.append(torch.matmul(P, query_feats[idx]).view(init_size))
            init_size = feature.shape
            feature = feature.view(bsz, C.size(2), -1)
            transformed_support_feats.append(torch.matmul(P, feature).view(init_size))

        return transformed_query_feats, transformed_support_feats

    def calDist(self, feature, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(feature, prototype[..., None, None], dim=1) * scaler
        return dist # dist:[1,53,53]

    def finetune_reference(self, batch, query_mask, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        kl_agg = 0
        cos_agg = 0
        for s_idx in range(nshot):
            support_feats_wo_mask = self.extract_feats(batch['support_imgs'][:, s_idx], self.backbone, self.feat_ids,
                                               self.bottleneck_ids, self.lids)
            support_feats, prototypes_sf, prototypes_sb = self.mask_feature(support_feats_wo_mask,
                                                                            batch['support_masks'][:, s_idx].clone())
            query_feats = self.extract_feats(batch['query_img'], self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            query_feats, prototypes_qf, prototypes_qb = self.mask_feature(query_feats, query_mask)

            kl = 0
            cos = 0

            for idx, feature in enumerate(prototypes_qf):
                if idx <= 3:
                    kl += F.kl_div(feature.softmax(dim=-1).log(), prototypes_sf[idx].softmax(dim=-1), reduction='sum')
            kl_agg += kl
            cos_agg += cos / 4

            if nshot == 1: return kl_agg

        return kl_agg/nshot


    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
