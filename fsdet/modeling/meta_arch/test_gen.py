from detectron2.utils.visualizer import Visualizer, _create_text_labels
import os
import matplotlib

novel_lsit=["bird", "bus", "cow", "motorbike", "sofa"]

def parse_rec( filename):
    import xml.etree.ElementTree as ET
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    bbox_list = []
    name = []
    for obj in tree.findall("object"):
        obj_struct = {}
        if(obj.find("name").text in novel_lsit):
            print(obj.find("name").text)
            continue
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
        bbox_list.append(obj_struct["bbox"])
        name.append(obj_struct["name"])
        objects.append(obj_struct)

    return bbox_list, name


def visualize_training( batched_inputs, root):
    """
    A function used to visualize images and proposals. It shows ground truth
    bounding boxes on the original image and up to 20 predicted object
    proposals on the original image. Users can implement different
    visualization functions for different models.

    Args:
        batched_inputs (list): a list that contains input to the model.
        proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
    """

    if not os.path.exists(root):
        os.mkdir(root)

    # storage = get_event_storage()
    max_vis_prop = 20
    # print('len_image  {}  len_pro  {}, type(por {}'.format(len(batched_inputs),len(proposals),type(proposals[0])))

    for input in batched_inputs:
        # img = input["image"]

        # print(prop)
        file_name = input # datasets/VOC2012/JPEGImages/2010_002656.jpg
        split_list=file_name.split('/')
        for ii in split_list:
            if 'voc_base' in ii:
                split_name=ii
            else:
                split_name='voc'
        print(file_name)
        anno_file = file_name.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')
        print(anno_file)
        print(file_name)
        bbox_list, name_list = parse_rec(anno_file)
        img = matplotlib.image.imread(file_name)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=bbox_list, labels=name_list)
        anno_img = v_gt.get_image()


        # h = anno_img.shape[0]
        # w = anno_img.shape[1] + prop_img.shape[1]
        #
        # vis_img = np.ones((h, w, 3))
        # vis_img = vis_img * 225
        # vis_img[:anno_img.shape[0], :anno_img.shape[1], :] = anno_img
        # vis_img[:prop_img.shape[0], anno_img.shape[1]:w, :] = prop_img
        file_name = input.split('/')[-1]
        iter=33
        vis_name = split_name+'__'+str(iter) + '_' + file_name
        print('vis_name  ',vis_name)
        # print('img.size={},img.shape={}'.format(vis_img.size,vis_img.shape))
        tmp_root = root +"/"+ vis_name
        print("root  ",tmp_root)
        # vis_img = np.array((vis_img - np.min(vis_img)) / (np.max(vis_img) - np.min(vis_img)))
        matplotlib.image.imsave(tmp_root, anno_img)


if __name__ == '__main__':
    img_list=[
              ]
    root='/hdd/master2/code/G-D/test_fsod/datasets/ab_af_oo'
    # os.makedirs(root)
    # img_root='/hdd/master2/code/G-D/test_fsod/datasets/voc_base1_3_test/JPEGImages'
    # for img in os.listdir(img_root):
    #     path=os.path.join(img_root,img)
    #     img_list.append(path)
    # img_list=['/hdd/master2/code/G-D/test_fsod/datasets/voc_base1_10/JPEGImages/2009_004249_gen.jpg',
    #           '/hdd/master2/code/G-D/test_fsod/datasets/voc_base1_3/JPEGImages/2008_007428_gen.jpg']
    # img_list=['/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/001543_000628_2007_gen.jpg',
    #           '/hdd/master2/code/G-D/test_fsod/datasets/voc_base3_10/JPEGImages/001543_004606_2007_gen.jpg',
    #           '/hdd/master2/code/G-D/test_fsod/datasets/voc_base3_10/JPEGImages/004129_2010_001630_2012_gen.jpg',
    #           '/hdd/master2/code/G-D/test_fsod/datasets/voc_base3_10/JPEGImages/006679_006352_2007_gen.jpg',
    #          '/hdd/master2/code/G-D/test_fsod/datasets/voc_base3_10/JPEGImages/005819_004195_2007_gen.jpg']
    img_list=['/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/008294.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/009845.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000102.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000104.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000232.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/003004.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000648.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000699.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/003021.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/003429.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/004069.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/006064.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/007358.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000025.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000116.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000108.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000386.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000384.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/000960.jpg',
              '/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/001031.jpg']
    img_list=['/hdd/master2/code/G-D/fsod/datasets/VOC2007/JPEGImages/008111.jpg']
    visualize_training(img_list,root)


    num_root='/hdd/master2/code/G-D/test_fsod/datasets/voc_base3_10/JPEGImages'
    num=0
    # for i in os.listdir(num_root):
    #     num=num+1
    print(num)
