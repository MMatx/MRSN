# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
import copy
import numpy as np

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from .build import PROPOSAL_GENERATOR_REGISTRY
from .proposal_utils import find_top_rpn_proposals

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_labels: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4,cfg:None):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        print('rpn  num_anchors ',num_anchors)
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
        self.cfg=cfg

        if cfg.SUPER_CLASSES_method=='after_conv2_parallel_log':
            self.super_class = torch.load(self.cfg.super_class_root)
            super_num=cfg.SUPER_CLASSES
            self.super_net_log=nn.ModuleList()
            for _ in range(super_num):
                self.super_net_log.append(nn.Conv2d(in_channels,num_anchors,kernel_size=1, stride=1))
            for sub_net in self.super_net_log:
                nn.init.normal_(sub_net.weight, std=0.01)
                nn.init.constant_(sub_net.bias, 0)

        elif cfg.SUPER_CLASSES_method=='after_backbone_parallel_conv2':
            self.super_class = torch.load(self.cfg.super_class_root)
            super_num=cfg.SUPER_CLASSES
            self.super_net_log = nn.ModuleList()
            self.super_net_con=nn.ModuleList()
            for _ in range(super_num):
                self.super_net_con.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
                self.super_net_log.append(nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1))
            for sub_net_log,sub_net_con in zip(self.super_net_log,self.super_net_con):
                nn.init.normal_(sub_net_log.weight, std=0.01)
                nn.init.constant_(sub_net_log.bias, 0)
                nn.init.normal_(sub_net_con.weight,std=0.01)
                nn.init.constant_(sub_net_con.bias,0)
        elif cfg.SUPER_CLASSES_method=='con_Similarity_MSE':
            self.super_class = torch.load(self.cfg.super_class_root)
            super_num = cfg.SUPER_CLASSES
            self.super_net_log = nn.ModuleList()
            for _ in range(super_num):
                self.super_net_log.append(nn.Conv2d(in_channels, 1, kernel_size=1, stride=1))
            for sub_net in self.super_net_log:
                nn.init.normal_(sub_net.weight, std=0.01)
                nn.init.constant_(sub_net.bias, 0)

        else:
            print('Not use Super_classes')

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim,'cfg':cfg}
    def enforce(self,x,super_class_f):
        # print(x.size(),super_class_f.size())
        #torch.Size([2, 256, 160, 216]) torch.Size([256])
        #[256,W,H] *[256]
        # enforce_x=
        super_class_f=super_class_f.to(x.device)
        super_class_f=super_class_f.view((1,256,1,1))
        enforce_x=x*super_class_f
        return enforce_x

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        # print('len(features)   ',len(features))
        #len(features)    5
# x   torch.Size([2, 256, 144, 240])
# x   torch.Size([2, 256, 72, 120])
# x   torch.Size([2, 256, 36, 60])
# x   torch.Size([2, 256, 18, 30])
# x   torch.Size([2, 256, 9, 15])

        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        if self.cfg.SUPER_CLASSES_method == 'after_conv2_parallel_log':
            ans_pred_objectness_logits=[]
            for idx,x in enumerate(features):
                t=F.relu(self.conv(x))
                max_cur_log=pred_objectness_logits[idx]
                super_idx=0
                for sub_net_log in self.super_net_log:
                    t = self.enforce(t,self.super_class[super_idx])
                    super_idx=super_idx+1
                    cur_log=sub_net_log(t)
                    max_cur_log=torch.max(max_cur_log,cur_log)
                ans_pred_objectness_logits.append(max_cur_log)
            return ans_pred_objectness_logits,pred_anchor_deltas

        elif self.cfg.SUPER_CLASSES_method == 'after_backbone_parallel_conv2':
            ans_pred_objectness_logits=[]
            for idx,x in enumerate(features):
                max_cur_log=pred_objectness_logits[idx]
                super_idx=0
                for sub_net_con,sub_net_log in zip(self.super_net_con,self.super_net_log):
                    t=self.enforce(x,self.super_class[super_idx])
                    super_idx=super_idx+1
                    t=sub_net_con(t)
                    cur_log=sub_net_log(t)
                    max_cur_log=torch.max(max_cur_log,cur_log)
                ans_pred_objectness_logits.append(max_cur_log)
            return ans_pred_objectness_logits,pred_anchor_deltas
        elif self.cfg.SUPER_CLASSES_method=='con_Similarity_MSE':
            simi_loss=torch.nn.MSELoss()
            ans_pred_objectness_logits = []
            #对每个超类都求一个相似性
            sum_super_smi_loss=[]
            for idx, x in enumerate(features):
                t = F.relu(self.conv(x))
                max_cur_log = pred_objectness_logits[idx]
                super_idx = 0
                for sub_net_log in self.super_net_log:
                    # t = self.enforce(t, self.super_class[super_idx])
                    super_class_f=self.super_class[super_idx].view(1,256,1,1)
                    super_class_f=super_class_f.to(t.device)
                    gt_smi=torch.cosine_similarity(t,super_class_f,dim=1)
                    super_idx = super_idx + 1
                    cur_log = sub_net_log(t)
                    max_cur_log = max_cur_log+cur_log*0.01
                    gt_smi=gt_smi.view(cur_log.size())
                    super_smi_loss=simi_loss(cur_log,gt_smi)
                    sum_super_smi_loss.append(super_smi_loss)
                ans_pred_objectness_logits.append(max_cur_log)
            return ans_pred_objectness_logits, pred_anchor_deltas,sum(sum_super_smi_loss)/len(sum_super_smi_loss)


        return pred_objectness_logits, pred_anchor_deltas

@RPN_HEAD_REGISTRY.register()
class Base2novelRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    求
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim}

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas
@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
        RPN_bg2unknow=0.0,
        RPN_bg2unknow_topk=0,
            cfg:None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of names of input features to use
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            batch_size_per_image (int): number of anchors per image to sample for training
            positive_fraction (float): fraction of foreground anchors to sample for training
            pre_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select before NMS, in
                training and testing.
            post_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select after NMS, in
                training and testing.
            nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            anchor_boundary_thresh (float): legacy option
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                    "loss_rpn_cls" - applied to classification loss
                    "loss_rpn_loc" - applied to box regression loss
            box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
            smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
                use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
        """
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.RPN_bg2unknow=RPN_bg2unknow
        self.RPN_bg2unknow_topk=RPN_bg2unknow_topk
        self.cfg=cfg

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        ret['RPN_bg2unknow']=cfg.RPN_bg2unknow
        ret['RPN_bg2unknow_topk']=cfg.RPN_bg2unknow_topk
        ret['cfg']=cfg
        return ret

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        old_gt_labels =[] #用于存储未采样的标签
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            # print('rpn 317 gt_labels_i before ',sum(gt_labels_i>=0))
            old_gt_labels.append(copy.deepcopy(gt_labels_i))
            # print('323 ', (old_gt_labels[-1] == 0).sum(), (gt_labels_i == 0).sum())
            # print('before gt_labels_i ',(gt_labels_i==0).sum())
            gt_labels_i = self._subsample_labels(gt_labels_i)
            # print('325 ',(old_gt_labels[-1]==0).sum(),(gt_labels_i==0).sum())
            # print('gt_labels_i after ',(gt_labels_i==0).sum())
            # print('rpn 317 gt_labels_i after ',sum(gt_labels_i>=0))
            #rpn 317 gt_labels_i before  tensor(216861, device='cuda:1')
            #rpn 317 gt_labels_i after  tensor(256, device='cuda:3')



            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            # print('341 ', (old_gt_labels[-1] == 0).sum(), (gt_labels[-1] == 0).sum())
            # print('rpn 335 ',type(gt_labels),type(old_gt_labels)) #rpn 335  <class 'list'> <class 'list'>
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes,old_gt_labels

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        old_gt_labels:List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        old_gt_labels=torch.stack(old_gt_labels)
        # print(' gt_labels ',(gt_labels==0).sum())
        # print('old_gt_labels ',(old_gt_labels==0).sum())
        valid_mask = gt_labels >= 0
        normalizer = self.batch_size_per_image * num_images
        storage = get_event_storage()
        # print('rpn 389 valid_mask ',valid_mask.size(),pred_objectness_logits[0].size())
        # rpn 389 valid_mask  torch.Size([4, 205923]) torch.Size([4, 154560])
        # rpn 389 valid_mask  torch.Size([4, 227124]) torch.Size([4, 170496])


        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()

        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)


        if self.box_reg_loss_type == "smooth_l1":
            anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)
            gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
            gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)
            localization_loss = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            pred_proposals = cat(pred_proposals, dim=1)
            pred_proposals = pred_proposals.view(-1, pred_proposals.shape[-1])
            pos_mask = pos_mask.view(-1)
            localization_loss = giou_loss(
                pred_proposals[pos_mask], cat(gt_boxes)[pos_mask], reduction="sum"
            )
        else:
            raise ValueError(f"Invalid rpn box reg loss type '{self.box_reg_loss_type}'")




        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )


        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }

        if self.RPN_bg2unknow > 0.0:
            print('rpn base1_RPN_bg2unknow_topk_25')
            # 增加模型对于novel类的识别能力，将标签是背景并且预测的置信度大于一定值的anchor，设置为1？
            # print('rpn 403 ',(pred_objectness_logits[0].max()),(pred_objectness_logits[0].min()),len(pred_objectness_logits))
            # print('rpn 407 gt_labels ', type(gt_labels), type(old_gt_labels), gt_labels.size(), old_gt_labels[0].size())

            #把置信度小于阈值的筛选掉，然后保留gt为0的框
            tmp_pred_objectness_logits=cat(pred_objectness_logits,dim=1) #torch.Size([4, 242991])
            # print(tmp_pred_objectness_logits.size(),type(tmp_pred_objectness_logits),type(pred_objectness_logits)) #([4, 196416])

            #先筛选置信度、
            # valid_mask_c=tmp_pred_objectness_logits<self.RPN_bg2unknow
            # print('rpn 388 valid_mask_c ',(tmp_pred_objectness_logits>=self.RPN_bg2unknow).sum())
            # print(old_gt_labels[tmp_pred_objectness_logits>=self.RPN_bg2unknow])
            # print(valid_mask[tmp_pred_objectness_logits>=self.RPN_bg2unknow])
            # print(gt_labels[tmp_pred_objectness_logits>=self.RPN_bg2unknow])
            # old_gt_labels[valid_mask_c]=-1 #筛选掉小与阈值的框
            # valid_mask_c_bg=old_gt_labels ==0 #选择为背景的框
            # print('rpn 391 old_gt_labels ==0 ',(old_gt_labels ==0).sum())
            # valid_mask_c_bg=(old_gt_labels==0) &(tmp_pred_objectness_logits>=self.RPN_bg2unknow)
            min_p=tmp_pred_objectness_logits.min()
            tmp_pred_objectness_logits[old_gt_labels!=0]=min_p
            v,indx=tmp_pred_objectness_logits.topk(self.RPN_bg2unknow_topk)
            thre=v.min()
            valid_mask_c_bg=(old_gt_labels==0)&(tmp_pred_objectness_logits>=thre)
            gt_labels[valid_mask_c_bg]=1
            #
            storage.put_scalar("rpn/before_sum_anchors", valid_mask_c_bg.sum())

            objectness_loss_bg = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[valid_mask_c_bg],
                gt_labels[valid_mask_c_bg].to(torch.float32),
                reduction="sum",
            )
            objectness_loss_bg=objectness_loss_bg*0.1/(valid_mask_c_bg.sum())
            # print('rpn 391 after ', (gt_labels >= 0).sum(),valid_mask.sum())
            storage.put_scalar("rpn/objectness_loss_bg", objectness_loss_bg)
            losses['objectness_loss_bg']=objectness_loss_bg
            # normalizer = valid_mask.sum()
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        # for k,v in features.items():
        #     print(k,'  ',v.size())
        #p2    torch.Size([4, 256, 200, 272])
# p3    torch.Size([4, 256, 100, 136])
# p4    torch.Size([4, 256, 50, 68])
# p5    torch.Size([4, 256, 25, 34])
# p6    torch.Size([4, 256, 13, 17])
# p2    torch.Size([4, 256, 168, 256])
# p3    torch.Size([4, 256, 84, 128])
# p4    torch.Size([4, 256, 42, 64])
# p5    torch.Size([4, 256, 21, 32])
# p6    torch.Size([4, 256, 11, 16])
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        # print('rpn   features  ',len(features),features[0].size())
        if self.cfg.SUPER_CLASSES_method=='con_Similarity_MSE':
            pred_objectness_logits, pred_anchor_deltas,super_smi_loss = self.rpn_head(features)
            # =
        else:

            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # print('pred_objectness_logits  ',len(pred_objectness_logits),'  ',pred_objectness_logits[0].size())
        # print('pred_anchor_deltas  ',len(pred_anchor_deltas),'    '  ,pred_anchor_deltas[0].size())
#         #rpn   features   5 torch.Size([4, 256, 192, 256])
        #rpn   features   5 torch.Size([4, 256, 184, 248])

# pred_objectness_logits   5    torch.Size([4, 3, 192, 256])
# pred_anchor_deltas   5      torch.Size([4, 12, 192, 256])


        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes ,old_gt_labels= self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes,old_gt_labels
            )
            if self.cfg.SUPER_CLASSES_method=='con_Similarity_MSE':
                # print('super_smi_loss  ',super_smi_loss)
                losses['con_Similarity_MSE']=super_smi_loss
                storage = get_event_storage()
                storage.put_scalar('rpn/super_smi_loss',super_smi_loss)
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        # print(losses)
        return proposals, losses

    # TODO: use torch.no_grad when torchscript supports it.
    # https://github.com/pytorch/pytorch/pull/41371
    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for approximate joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses, so is approximate.
        pred_objectness_logits = [t.detach() for t in pred_objectness_logits]
        pred_anchor_deltas = [t.detach() for t in pred_anchor_deltas]
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            # https://github.com/pytorch/pytorch/issues/41449
            self.pre_nms_topk[int(self.training)],
            self.post_nms_topk[int(self.training)],
            self.min_box_size,
            self.training,
        )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        # print('RPN  _decode_proposals')
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals
