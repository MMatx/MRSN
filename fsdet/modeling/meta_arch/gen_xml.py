import os
import xml
import xml.etree.ElementTree as ET
from lxml.etree import Element,SubElement,tostring
import pprint
from xml.dom.minidom import parseString
import cv2
from PIL import Image
import tqdm

def f_read_xml(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)
    return objects
def f_save_xml(width,height,image,split_object_ans,save_xml):
    # 创建文件的头信息
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC20077'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image
    #####################################################################
    # 图片的长宽
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    #####################################################################
    # 物体
    for sub_obj in split_object_ans:
        # print(sub_obj)
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        # print("type sub_obj['name'] ",sub_obj['name'],type(sub_obj['name']))
        node_name.text = sub_obj['name']
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Left'
        node_truncates = SubElement(node_object, 'truncated')
        node_truncates.text = '1'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(sub_obj['bbox'][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(sub_obj['bbox'][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(sub_obj['bbox'][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(sub_obj['bbox'][3])
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    with open(save_xml, 'w', encoding='UTF-8') as fh:
        dom.writexml(fh, indent='', addindent='\t', encoding='UTF-8')
# root='VOC2012'
# save_root='VOC20122'
# if not os.path.exists(save_root):
#     os.mkdir(save_root)
# id='2' #2或者7，为了给后面生成的文件做标记  2007：7 2012:2
# split='train'  #用于标记是train还是val
# root_txt='/home/mtx/fine-tune-fsod/code/datasets/VOC2012/ImageSets/Main/train.txt'
# c_root=['Annotations', 'JPEGImages', 'ImageSets']
# for i in c_root:
#     if not os.path.exists(root+'/'+i):
#         # print('i  ',i)
#         os.mkdir(root+'/'+i)
#
#
#
# old_image_root=root+'/'+c_root[1]
# save_image_root=save_root+'/'+c_root[1]
# if not os.path.exists(save_image_root):
#     os.mkdir(save_image_root)
# old_xml_root=root+'/'+c_root[0]
# save_xml_root=save_root+'/'+c_root[0]
# if not os.path.exists(save_xml_root):
#     os.mkdir(save_xml_root)
# if not os.path.exists(save_root+'/'+c_root[2]):
#     os.mkdir(save_root+'/'+c_root[2])
# old_txt_root=root+'/'+c_root[2]+'/'+'Main'
# save_txt_root=save_root+'/'+c_root[2]+'/'+'Main'
# if not os.path.exists(save_txt_root):
#     os.mkdir(save_txt_root)
# class_name_train=[]
# class_name_val=[]
# class_name_test=[]
#
# ##### 生成txt文件的文件名
# for class_txt in os.listdir(old_txt_root):
#     new_txt=save_txt_root+'/'+class_txt
#     if not os.path.exists(save_txt_root):
#         os.mkdir(save_txt_root)
#     if not os.path.exists(new_txt):
#         os.mknod(new_txt)
#
# for f in os.listdir(save_txt_root):
#     if split in new_txt:
#         if f=='test.tx' or f=='train.txt' or f=='trainval.txt' or f=='val.txt':
#             continue
#         if split == 'train' and 'train' in f:
#             if f=='train_val.txt' or f=='train_test.txt':
#                 continue
#             class_name_train.append(f)
#         elif split == 'val' and 'val' in f:
#             class_name_val.append(f)
#         elif split == 'test' and 'test' in f :
#             class_name_test.append(f)

#切图规则
#1、面积<thera:不切
#2、

# def gen_obj_struct(name,pose,truncated,difficult,xmin,ymin,xmax,ymax):
#     obj_struct = {}
#     obj_struct["name"] = name
#     obj_struct["pose"] = pose
#     obj_struct["truncated"] = truncated
#     obj_struct["difficult"] = difficult
#     obj_struct["bbox"] = [
#         xmin,
#         ymin,
#         xmax,
#         ymax,
#     ]
#     return obj_struct
# def split_image(old_image,ob,objects,):
#     #输入
#     split_ans=[]
#     img_1 = Image.open(old_image)
#     old_width,old_height=img_1.size[0],img_1.size[1]
#     obj_xmin,obj_ymin,obj_xmax,obj_ymax=ob['bbox']
#     cent_x=abs(obj_xmax-obj_xmin)//2+obj_xmin
#     cent_y=abs(obj_ymax-obj_ymin)//2+obj_ymin
#     xmin_x=[0,cent_x,0,cent_x]
#     ymin_y=[0,0,cent_y,cent_y]
#     xmax_x=[cent_x,old_width,cent_x,old_width]
#     ymax_y=[cent_y,cent_y,old_height,old_height]
#     for i in range(4):
#         sub_image={}
#         cur_xmin=xmin_x[i]
#         cur_ymin=ymin_y[i]
#         cur_xmax=xmax_x[i]
#         cur_ymax=ymax_y[i]
#
#         cur_objects=[]
#
#         for obb in objects:
#             other_xmin,other_ymin,other_xmax,other_ymax=obb['bbox']
#             U_xmin=max(cur_xmin,other_xmin)
#             U_ymin=max(cur_ymin,other_ymin)
#             U_xmax=min(cur_xmax,other_xmax)
#             U_ymax=min(cur_ymax,other_ymax)
#             if U_xmin>U_xmax:
#                 continue
#             if U_ymin>U_ymax:
#                 continue
#             #####要使用相对坐标
#
#             re_xmin=U_xmin-cur_xmin
#             re_xmax=U_xmax-cur_xmin
#             re_ymin=U_ymin-cur_ymin
#             re_ymax=U_ymax-cur_ymin
#             if (abs(re_xmax-re_xmin)*abs(re_ymax-re_ymin)<16*16):
#                 continue
#             cur_objects.append(gen_obj_struct(obb['name'],obb['pose'],obb['truncated'],obb['difficult'],re_xmin,re_ymin,re_xmax,re_ymax))
#         sub_image['xmin']=cur_xmin
#         sub_image['ymin']=cur_ymin
#         sub_image['xmax']=cur_xmax
#         sub_image['ymax']=cur_ymax
#         sub_image['width']=abs(cur_xmax-cur_xmin)
#         sub_image['height']=abs(cur_ymax-cur_ymin)
#         sub_image['ans_object']=cur_objects
#         split_ans.append(sub_image)
#
#     return split_ans

# print(trainval)
# def f_save_image(old_image,image,xmin,ymin,xmax,ymax):
#     # python裁剪图片并保存
#     # 读取图片
#     img_1 = Image.open(old_image)
#     # 设置裁剪的位置
#     crop_box = (xmin, ymin, xmax,ymax)  # xmin,ymin,xmax,ymax
#     # 裁剪图片
#     img_2 = img_1.crop(crop_box)
#     crop_save_path=save_image_root+'/'+image
#     img_2.save(crop_save_path)


# def f_save_txt(save_txt_root,ans_object,image):
#     image=image.split('.')[0]
#     # print('image ',image)
#     cl_txt_name=[]
#     # print(split)
#     # print('class_name_train  ',class_name_train)
#     if split=='train':
#         cl_txt_name=class_name_train
#     elif split=='val':
#         cl_txt_name=class_name_val
#     have_cl=[]
#     for sub_obj in ans_object:
#         have_cl.append(sub_obj['name'])
#     # print('have_cl  ',have_cl)
#     # print('cl_txt_name ',cl_txt_name,len(cl_txt_name))
#     for cl_path in cl_txt_name:
#         txt_path = save_txt_root + '/' + cl_path
#         # print('txt_path  ',txt_path)
#         with open(txt_path,'a') as f:
#             cur_txt_cls=cl_path.split('_')[0]
#             # print(cur_txt_cls)
#             strr=''
#             if cur_txt_cls in have_cl:
#                 strr=image+'  '+str(1)+'\n'
#             else:
#                 strr=image+'  '+str(0)+'\n'
#             # print(strr)
#             f.write(strr)
#             f.close()

# with open(root_txt,'r') as f:
#     num=0
#     while True:
#         name=f.readline()
#         if not name:
#             break
#         name=name.replace('\n','')
#         num=num+1
#         # if num>10:  #先切分几个，小批量处理处理，方便调试
#         #     break
#         old_xml=old_xml_root+'/'+name+'.xml'
#         old_image=old_image_root+'/'+name+'.jpg'
#         objects=read_xml(old_xml)
#         # print(object)
#         #以每张图片中的每一个物体为中心，进行切分
#         show=0   #对切分结果进行可视化展示
#         sun_obj=1
#         if show:
#             save_show_root='/home/mtx/fine-tune-fsod/code/datasets/test_crop_show'
#             image_whole=cv2.imread(old_image)
#             for ob in objects:
#                 cv2.rectangle(image_whole,(ob['bbox'][0],ob['bbox'][1]),(ob['bbox'][2],ob['bbox'][3]),(0,255,0),2)
#             if not os.path.exists(save_show_root+'/'+name):
#                 os.mkdir(save_show_root+'/'+name)
#             cv2.imwrite(save_show_root+'/'+name+'/'+'whole.jpg', image_whole)
#
#
#
#         for whole_image_ob_idx,ob in enumerate(objects):  #图中的每个instance都会当一次中心处理点
#             ##backbone下采样16，面积＜16*16的物体不进行切分
#             if(whole_image_ob_idx%50==0):
#                 print('{}/{}'.format(whole_image_ob_idx,len(objects)))
#             if int(ob['difficult'])==1:
#                 continue
#             area=(abs(int(ob['bbox'][2])-int(ob['bbox'][0]))*(abs(int(ob['bbox'][3])-int(ob['bbox'][1]))))
#             if (16*16)>area:
#                 continue
#             ####开始切分，切分结果
#             split_ans=split_image(old_image,ob,objects)
#             for cur_ob_pathch,sub_split_ans in enumerate(split_ans): #以当前instance为中心处理点，会产生几个子图
#                 xmin, ymin, xmax, ymax, width, height, ans_object =sub_split_ans['xmin'],sub_split_ans['ymin'],sub_split_ans['xmax'],sub_split_ans['ymax'],sub_split_ans['width'],sub_split_ans['height'],sub_split_ans['ans_object']
#                 ########保存的地址和信息
#                 save_xml = save_xml_root + '/' + name + '_' + str(id) + '_' + str(whole_image_ob_idx)+'_'+str(cur_ob_pathch) + '.xml'
#                 image = name + '_' + str(id) + '_' + str(whole_image_ob_idx) +'_'+str(cur_ob_pathch)+ '.jpg'
#                 save_image = save_image_root + '/' + name + '_' + str(id) + '_' + str(whole_image_ob_idx) +'_'+str(cur_ob_pathch)+ '.jpg'
#                 #####################################################################
#                 f_save_xml(width, height, image, ans_object, save_xml)
#                 f_save_image(old_image, image, xmin, ymin, xmax, ymax)
#                 f_save_txt(save_txt_root, ans_object, image)
#                 save_txt_con=name + '_' + str(id) + '_' + str(whole_image_ob_idx) +'_'+str(cur_ob_pathch)
#                 if split=='train':
#                     with open(save_txt_root+'/'+'train.txt','a') as train_f:
#                         train_f.write(save_txt_con+'\n')
#                         train_f.close()
#                     with open(save_txt_root+'/'+'trainval.txt','a') as train_val_f:
#                         train_val_f.write(save_txt_con+'\n')
#                         train_val_f.close()
#                 elif split=='val':
#                     with open(save_txt_root+'/'+'val.txt','a')as val_f:
#                         val_f.write(save_txt_con+'\n')
#                         val_f.close()
#                     with open(save_txt_root+'/'+'trainval.txt','a') as train_val_f:
#                         train_val_f.write(save_txt_con+'\n')
#                         train_val_f.close()
#                 elif split=='test':
#                     with open(save_txt_root+'/'+'test.txt','a')as test_f:
#                         test_f.write(save_txt_con+'\n')
#                         test_f.close()
#                 if show:
#                     # save_show_root =
#                     r=save_image
#                     # print(r)
#                     image_sub = cv2.imread(r)
#                     for ob in ans_object:
#                         cv2.rectangle(image_sub, (ob['bbox'][0], ob['bbox'][1]), (ob['bbox'][2], ob['bbox'][3]),
#                                       (0, 255, 0), 2)
#                     s=save_show_root + '/' + name+'/'+str(whole_image_ob_idx)
#                     # print(s)
#                     # print(s +'/'+str(cur_ob_pathch) + '_sub.jpg')
#                     # print(image_sub.size)
#                     if not os.path.exists(s ):
#                         os.mkdir(s )
#                     cv2.imwrite(s +'/'+str(cur_ob_pathch) + '_sub.jpg', image_sub)
#
#
#     f.close()
