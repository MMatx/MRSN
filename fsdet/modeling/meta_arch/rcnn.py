import torch
from torch import nn
import os
import random
import numpy as np

from fsdet.modeling.roi_heads import build_roi_heads
from detectron2.data import MetadataCatalog
import logging
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.modeling.proposal_generator import build_proposal_generator
from fsdet.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from fvcore.common.file_io import PathManager
# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY
from .copy_and_paste import CopyAndPaste
from .gen_xml import f_save_xml,f_read_xml
from PIL import Image
import mmcv
__all__ = ["GeneralizedRCNN", "ProposalNetwork"]
import xml.etree.ElementTree as ET

def read_xml(root_ann):
    tree = ET.parse(root_ann)
    instances= {}
    for obj in tree.findall("object"):
        cls_ = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox = [
            float(bbox.find(x).text)
            for x in ["xmin", "ymin", "xmax", "ymax"]
        ]
        bbox[0] -= 1.0
        bbox[1] -= 1.0
        if cls_ not in instances:
            instances[cls_]=[]

        instances[cls_].append(bbox)
    return instances
def check_root(save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_root_img = save_root + '/' + 'JPEGImages'
    save_root_ann = save_root + '/' + 'Annotations'
    save_root_Set = save_root + '/' + 'ImageSets'
    save_root_txt = save_root_Set + '/' + 'Main'
    txt_name = 'train.txt'
    if not os.path.exists(save_root_img):
        os.mkdir(save_root_img)
    if not os.path.exists(save_root_ann):
        os.mkdir(save_root_ann)
    if not os.path.exists(save_root_Set):
        os.mkdir(save_root_Set)
    if not os.path.exists(save_root_txt):
        os.mkdir(save_root_txt)
    save_root_txt=os.path.join(save_root_txt,txt_name)
    return save_root_img,save_root_ann,save_root_txt
def get_novel_img_ann(data_name,split_id):
    dataset_root = '/hdd/master2/code/G-D/fsod'
    meta = MetadataCatalog.get(data_name)
    split_id = split_id
    novel_txt_root = '/hdd/master2/code/G-D/fsod/datasets/vocsplit'
    novel_class = meta.novel_classes
    use_novel_id = random.randint(0, len(novel_class) - 1)
    use_novel_class = novel_class[use_novel_id]
    read_txt_root = novel_txt_root + '/' + 'box_' + str(split_id[1]) + 'shot_' + use_novel_class + '_train.txt'
    with PathManager.open(
            read_txt_root
    ) as f:
        fileids_ = np.loadtxt(f, dtype=np.str).tolist()
        if isinstance(fileids_, str):
            fileids_ = [fileids_]
    # 取需要裁剪的novel类的id
    use_image_id = random.randint(0, len(fileids_) - 1)
    novel_img_root = os.path.join(dataset_root, fileids_[use_image_id])
    # print('novel_img_root  ', novel_img_root)
    novel_xml_root = novel_img_root.replace('jpg', 'xml').replace('JPEGImages', 'Annotations')
    use_image_name=fileids_[use_image_id].split('/')[-1].split('.')[0]
    return novel_img_root,novel_xml_root,use_image_name,use_novel_class
class Stack(object):

    def __init__(self):
        self.stack = []

    def push(self, data):
        """
        进栈函数
        """
        self.stack.append(data)

    def pop(self):
        """
        出栈函数，
        """
        return self.stack.pop()

    def gettop(self):
        """
        取栈顶
        """
        return self.stack[-1]
    def empty(self):
        if len(self.stack)==0:
            return true
        else:
            return false
def find_bg():
    all_img=np.zeros((n,m))
    file_xml=file_name.replace('jpg', 'xml').replace('JPEGImages', 'Annotations')
    obs=read_xml(file_xml)
    for ob_name,ob_box in obs.items():
        all_img[ob_box[0]:ob_box[2],ob_box[1]:ob_box[3]]=1
    up=np.zeros((1,m))
    stack = Stack()
    for i in range(n):
        for j in range(m):
            if all_img==1:
                up[j]=0
            else:
                up[j]=1
            while not stack.empty():
                stack.pop()
            for j in range(m):
                while not stack.empty():
                    if up[stack.gettop()>up[j]]:
                        tmp=stack.gettop()
                        stack.pop()
                        ans=max(ans,up[tmp]*(j-stack.gettop()-1))
                stack.push(j)




def trans_img(file_name,crop_novel_img,use_image_name):

    i=random.randint(1,20)
    if i==5:
        crop_novel_img=mmcv.imflip(crop_novel_img) #crop_novel_img.transpose(Image.FLIP_LEFT_RIGHT)
        name1=use_image_name+'_crop_flip.jpg'
        # mmcv.imwrite(crop_novel_img,'/hdd/master2/code/G-D/test_fsod/dataset/test/'+name1)
    bg_img=mmcv.imread(file_name)
    name1 = use_image_name + '_base.jpg'
    # mmcv.imwrite(bg_img,'/hdd/master2/code/G-D/test_fsod/dataset/test/' + name1)

    CP=CopyAndPaste()
    bg_img,xmin,xmax, ymin,ymax=CP.c_and_p(bg_img,crop_novel_img)
    name1 = use_image_name + '_paste.jpg'
    # mmcv.imwrite(bg_img,'/hdd/master2/code/G-D/test_fsod/dataset/test/' + name1)
    return bg_img, ymin,ymax,xmin,xmax




@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.sum_all_num=0
        self.save_num=0

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")
        if cfg.freeze_cluster_net_log:
            for name,p in self.proposal_generator.rpn_head.super_net_log.named_parameters():
                print('freeze_cluster_net')
                print(name)
                p.requires_grad=False
        if cfg.freeze_cluster_net_con:
            for name,p in self.proposal_generator.rpn_head.super_net_con.named_parameters():
                print(name)
                p.requires_grad=False
            print('freeze_cluster_net_con')
        self.cfg=cfg
        self.have_save_num=0
        self.all_img_num=0
        self.use_img_num=0
        self.res_save_img=0
        self.data_voc=True

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training and self.data_voc:
            res_tmp=self.inference(batched_inputs)
            # print('rcnn 104  ',res_tmp)
            dataset_root='/hdd/master2/code/G-D/fsod'


            for img,res in zip(batched_inputs,res_tmp):
                self.all_img_num = self.all_img_num + 1
                file_name=img['file_name']
                img_id=img['image_id']
                pre_class=res['instances'].pred_classes
                mask=pre_class>15
                if(sum(mask)==0):
                    self.use_img_num = self.use_img_num + 1

                    save_root=self.cfg.OUTPUT_DIR
                    data_name=self.cfg.DATASETS.TEST[0]
                    meta = MetadataCatalog.get(data_name)
                    ###合成img保存路径，图片保存路径，注释保存路径，全部训练图片的txt文件，
                    save_root_img,save_root_ann,save_root_txt=check_root(save_root)
                    ###获取需要增强的novel类
                    novel_img_root,novel_xml_root,use_image_name,use_novel_class=get_novel_img_ann(data_name,self.cfg.split_id)
                    ####裁剪对应区域
                    novel_img = mmcv.imread(novel_img_root)
                    instances = read_xml(novel_xml_root)
                    name1 = use_image_name + '_old.jpg'
                    # mmcv.imwrite(novel_img,'/hdd/master2/code/G-D/test_fsod/dataset/test/' + name1)
                    crop_novel_img = mmcv.imcrop(novel_img,np.array([
                        instances[use_novel_class][0][0], instances[use_novel_class][0][1],
                        instances[use_novel_class][0][2],
                        instances[use_novel_class][0][3]
            ]))
                    name1 = use_image_name + '_crop.jpg'
                    # mmcv.imwrite(crop_novel_img, '/hdd/master2/code/G-D/test_fsod/dataset/test/' + name1)
                    ####对裁剪出的novel类区域进行增强,将novel类贴在base数据集上
                    gen_img,xmin,xmax, ymin,ymax=trans_img(file_name,crop_novel_img,use_image_name)
                    base_xml_path=file_name.replace('jpg', 'xml').replace('JPEGImages', 'Annotations')
                    objects=f_read_xml(base_xml_path)
                    new_objs=[]
                    obj_struct = {}
                    obj_struct["name"] = use_novel_class
                    obj_struct["bbox"] = [
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax),
                    ]
                    print((ymax-ymin)*(xmax-xmin))
                    if((ymax-ymin)*(xmax-xmin)<32):
                        continue
                    self.res_save_img=self.res_save_img+1
                    print('all_img_num={},use_img_num={},res_save_img={}'.format(self.all_img_num, self.use_img_num,self.res_save_img))

                    # objects.append(obj_struct)
                    new_objs.append(obj_struct)
                    for ob in objects:
                        if ob['name'] in meta.novel_classes:
                            continue
                        else:
                            new_objs.append(ob)
                    base_name=img_id
                    # print('file_name  ',file_name)
                    if '2007' in file_name:
                        gen_img_name = use_image_name + '_' + base_name + '_2007_gen'
                    else:
                        gen_img_name = use_image_name + '_' + base_name + '_2012_gen'



                    save_gen_xml=gen_img_name+'.xml'
                    save_gen_img=gen_img_name+'.jpg'
                    f_save_xml(img['width'], img['height'], gen_img_name+'.jpg', new_objs, save_root_ann+'/'+save_gen_xml)
                    mmcv.imwrite(gen_img,save_root_img+'/'+save_gen_img)

                    with open(save_root_txt,'a') as f:
                        ss=gen_img_name+'\n'
                        f.write(ss)

                    show_root=use_image_name + '_' + base_name
                    if not os.path.exists(show_root):
                        os.makedirs(os.path.join(save_root,show_root))
                    mmcv.imwrite(crop_novel_img,show_root+'/'+'crop.png')
                    bg_img = mmcv.imread(file_name)
                    mmcv.imwrite(bg_img, show_root + '/' + 'base.png')
                    mmcv.imwrite(gen_img, show_root + '/' + 'com.png')
                    print(self.have_save_num)
                    self.have_save_num=self.have_save_num+1


            return res_tmp
        elif not self.training and self.data_voc==False:
            res_tmp = self.inference(batched_inputs)
            # print('rcnn 104  ',res_tmp)
            dataset_root = '/hdd/master2/code/G-D/fsod'

            for img, res in zip(batched_inputs, res_tmp):
                self.all_img_num = self.all_img_num + 1
                file_name = img['file_name']
                img_id = img['image_id']
                pre_class = res['instances'].pred_classes
                mask = pre_class > 60
                if (sum(mask) == 0):
                    self.use_img_num = self.use_img_num + 1

                    save_root = self.cfg.OUTPUT_DIR
                    data_name = self.cfg.DATASETS.TEST[0]
                    meta = MetadataCatalog.get(data_name)
                    print(meta)
                    ###合成img保存路径，图片保存路径，注释保存路径，全部训练图片的txt文件，
                    save_root_img, save_root_ann, save_root_txt = check_root(save_root)
                    ###获取需要增强的novel类
                    novel_img_root, novel_xml_root, use_image_name, use_novel_class = get_novel_img_ann(data_name,
                                                                                                        self.cfg.split_id)
                    ####裁剪对应区域
                    novel_img = mmcv.imread(novel_img_root)
                    instances = read_xml(novel_xml_root)
                    name1 = use_image_name + '_old.jpg'
                    # mmcv.imwrite(novel_img,'/hdd/master2/code/G-D/test_fsod/dataset/test/' + name1)
                    crop_novel_img = mmcv.imcrop(novel_img, np.array([
                        instances[use_novel_class][0][0], instances[use_novel_class][0][1],
                        instances[use_novel_class][0][2],
                        instances[use_novel_class][0][3]
                    ]))
                    name1 = use_image_name + '_crop.jpg'
                    # mmcv.imwrite(crop_novel_img, '/hdd/master2/code/G-D/test_fsod/dataset/test/' + name1)
                    ####对裁剪出的novel类区域进行增强,将novel类贴在base数据集上
                    gen_img, xmin, xmax, ymin, ymax = trans_img(file_name, crop_novel_img, use_image_name)
                    base_xml_path = file_name.replace('jpg', 'xml').replace('JPEGImages', 'Annotations')
                    objects = f_read_xml(base_xml_path)
                    new_objs = []
                    obj_struct = {}
                    obj_struct["name"] = use_novel_class
                    obj_struct["bbox"] = [
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax),
                    ]
                    print((ymax - ymin) * (xmax - xmin))
                    if ((ymax - ymin) * (xmax - xmin) < 32):
                        continue
                    self.res_save_img = self.res_save_img + 1
                    print('all_img_num={},use_img_num={},res_save_img={}'.format(self.all_img_num, self.use_img_num,
                                                                                 self.res_save_img))

                    # objects.append(obj_struct)
                    new_objs.append(obj_struct)
                    for ob in objects:
                        if ob['name'] in meta.novel_classes:
                            continue
                        else:
                            new_objs.append(ob)
                    base_name = img_id
                    # print('file_name  ',file_name)
                    if '2007' in file_name:
                        gen_img_name = use_image_name + '_' + base_name + '_2007_gen'
                    else:
                        gen_img_name = use_image_name + '_' + base_name + '_2012_gen'

                    save_gen_xml = gen_img_name + '.xml'
                    save_gen_img = gen_img_name + '.jpg'
                    f_save_xml(img['width'], img['height'], gen_img_name + '.jpg', new_objs,
                               save_root_ann + '/' + save_gen_xml)
                    mmcv.imwrite(gen_img, save_root_img + '/' + save_gen_img)
                    with open(save_root_txt, 'a') as f:
                        ss = gen_img_name + '\n'
                        f.write(ss)
            return res_tmp
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = {}
        # print('detector_losses  rcnn 140 ',detector_losses)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )
        # print('rcnn 186  result ',type(results))
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
