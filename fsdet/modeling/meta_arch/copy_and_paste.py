import base64
import math
import random

import cv2
import mmcv
import numpy as np

# from ..builder import PIPELINES


# @PIPELINES.register_module()
class CopyAndPaste(object):

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 base_scale=(1333, 800),
                 scale=(0.2, 1.),
                 ratio=(3. / 4., 4. / 3.),
                 w_range=(100, 1000),
                 h_range=(100, 1000),
                 feed_bytes=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None if feed_bytes else mmcv.FileClient(
            **self.file_client_args)
        self.base_scale = base_scale
        self.scale = scale
        self.ratio = ratio
        self.w_range = w_range  # TODO: modify the range
        self.h_range = h_range
        self.feed_bytes = feed_bytes

    # @staticmethod
    def get_rescale_param(self,img_base,img_novel, ratio, scale=(0.2, 1.0)):
        # height = img.shape[0]
        # width = img.shape[1]
        # area = height * width
        base_h=img_base.shape[1]
        base_w=img_base.shape[0]
        base_area=base_h*base_w

        novel_h=img_novel.shape[1]
        novel_w=img_novel.shape[0]
        novel_area=novel_h*novel_w
        # per_novel_base=novel_area*1.0/base_area
        target_area_size=0.25*base_area
        target_scale=target_area_size/novel_area
        scale=(0.2,min(1.0,target_scale))

        for _ in range(20):
            target_area = random.uniform(*scale) * novel_area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= base_w and 0 < target_height <= base_h:
                return target_height, target_width
                # xmin = random.randint(0, height - target_height)
                # ymin = random.randint(0, width - target_width)
                # return xmin, ymin, target_height, target_width
        return 1,1
        # Fallback to central crop
        # in_ratio = float(width) / float(height)
        # if in_ratio < min(ratio):
        #     target_width = width
        #     target_height = int(round(target_width / min(ratio)))
        # elif in_ratio > max(ratio):
        #     target_height = height
        #     target_width = int(round(target_height * max(ratio)))
        # else:  # whole image
        #     target_width = width
        #     target_height = height
        # xmin = (height - target_height) // 2
        # ymin = (width - target_width) // 2
        # return xmin, ymin, target_height, target_width

    @staticmethod
    def get_position_param(crop, size=(256, 256)):
        W, H = size[0], size[1]
        w, h, _ = crop.shape
        sampled_w = int(np.random.randint(0, max(W - w,1), 1))
        sampled_h = int(np.random.randint(0, max(H - h,1), 1))
        return sampled_w, sampled_h


    def c_and_p(self,bg_img,crop_novel_img):
        base_img0 = bg_img
        # print(type(base_img0))
        # print('base_img0.size  ',base_img0.shape)

        W, H,C= base_img0.shape
        # print(W,H,C)
        BASE_SIZE = (W, H)
        q_img = base_img0.copy()
        # k_img = base_img1.copy()

        # num_imgs = len(chosen_img)

        # Get crop
        # crop1 = self.get_crop(chosen_img, self.ratio)
        crop1=crop_novel_img

        # Get scale
        # sampled_scale1 = self.get_scale()
        target_h,targe_w=self.get_rescale_param(base_img0,crop1, self.ratio)
        # sampled_scale2 = self.get_scale()

        # Rescale foreground images
        # print('before crop1  ',crop1.shape)
        try:
            crop1 = mmcv.imrescale(crop1, (targe_w,target_h))
        except:
            crop1 = mmcv.imresize(crop1, (targe_w,target_h))
        # print('adter crop  ',crop1.shape)

        # Sample Location
        sampled_w, sampled_h, _ = crop1.shape
        position_w, position_h = self.get_position_param(crop1, BASE_SIZE)
        q_img[position_w:position_w + sampled_w,
        position_h:position_h + sampled_h,:] = crop1
        return q_img,position_w,position_w + sampled_w, position_h,position_h + sampled_h
