#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__', '001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross','004.Groove_billed_Ani',
                         '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet',
                         '009.Brewer_Blackbird', '010.Red_winged_Blackbird', '011.Rusty_Blackbird','012.Yellow_headed_Blackbird',
                         '013.Bobolink', '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Buntings',
                         '017.Cardinal', '018.Spotted_Catbird', '019.Gray_Catbird', '020.Yellow_breasted_Chat',
                         '021.Eastern_Towhee', '022.Chuck_will_Widow', '023.Brandt_Cormorant','024.Red_faced_Cormorant',
                         '025.Pelagic_Cormorant', '026.Bronzed_Cowbird', '027.Shiny_Cowbirds', '028.Brown_Creeper',
                         '029.American_Crow', '030.Fish_Crow', '031.Black_billed_Cuckoo', '032.Mangrove_Cuckoo',
                         '033.Yellow_billed_Cuckoo', '034.Gray_crowned_Rosy_Finch', '035.Purple_Finch','036.Northern_Flicker',
                         '037.Acadian_Flycatcher', '038.Great_Crested_Flycatcher', '039.Least_Flycatcher','040.Olive_sided_Flycatcher',
                         '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '043.Yellow_bellied_Flycatcher','044.Frigatebird',
                         '045.Northern_Fulmar', '046.Gadwall', '047.American_Goldfinch', '048.European_Goldfinch',
                         '049.Boat_tailed_Grackle', '050.Eared_Grebe', '051.Horned_Grebe', '052.Pied_billed_Grebe',
                         '053.Western_Grebe', '054.Blue_Grosbeak', '055.Evening_Grosbeak', '056.Pine_Grosbeak',
                         '057.Rose_breasted_Grosbeak', '058.Pigeon_Guillemot', '059.California_Gull','060.Glaucous_winged_Gull',
                         '061.Heermann_Gull', '062.Herring_Gull', '063.Ivory_Gull', '064.Ring_billed_Gull',
                         '065.Slaty_backed_Gull', '066.Western_Gull', '067.Anna_Hummingbird','068.Ruby_throated_Hummingbird'
                         '069.Rufous_Hummingbird', '070.Green_Violetear', '071.Long_tailed_Jaeger','072.Pomarine_Jaeger',
                         '073.Blue_Jay', '074.Florida_Jay', '075.Green_Jay', '076.Dark_eyed_Junco',
                         '077.Tropical_Kingbird', '078.Gray_Kingbird', '079.Belted_Kingfisher', '080.Green_Kingfisher',
                         '081.Pied_Kingfisher', '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher', '084.Red_legged_Kittiwake',
                         '085.Horned_Lark', '086.Pacific_Loon', '087.Mallard', '088.Western_Meadowlark',
                         '089.Hooded_Merganser', '090.Red_breasted_Merganser', '091.Mockingbird', '092.Nighthawk',
                         '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole','096.Hooded_Oriole',
                         '097.Orchard_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelicansss',
                         '101.White_Pelican', '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit',
                         '105.Whip_poor_Will', '106.Horned_Puffin', '107.Common_Raven', '108.White_necked_Raveni',
                         '109.American_Redstart', '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike',
                         '113.Baird_Sparrow', '114.Black_throated_Sparrow', '115.Brewer_Sparrow', '116.Chipping_Sparrow',
                         '117.Clay_colored_Sparrow', '118.House_Sparrow', '119.Field_Sparrow', '120.Fox_Sparrow',
                         '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '124.Le_Conte_Sparrow',
                         '125.Lincoln_Sparrow', '126.Nelson_Sharp_tailed_Sparrow', '127.Savannah_Sparrow','128.Seaside_Sparrow',
                         '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow', '132.White_crowned_Sparrow',
                         '133.White_throated_Sparrow', '134.Cape_Glossy_Starling', '135.Bank_Swallow','136.Barn_Swallow',
                         '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager', '140.Summer_Tanager',
                         '141.Artic_Tern', '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern',
                         '145.Elegant_Tern', '146.Forsters_Tern', '147.Least_Tern', '148.Green_tailed_Towhee',
                         '149.Brown_Thrasher', '150.Sage_Thrasher', '151.Black_capped_Vireo', '152.Blue_headed_Vireo',
                         '153.Philadelphia_Vireo', '154.Red_eyed_Vireo', '155.Warbling_Vireo', '156.White_eyed_Vireo',
                         '157.Yellow_throated_Vireo', '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler','160.Black_throated_Blue_Warbler',
                         '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbleri',
                         '165.Chestnut_sided_Warbler', '166.Golden_winged_Warbler', '167.Hooded_Warbler','168.Kentucky_Warbler',
                         '169.Magnolia_Warbler', '170.Mourning_Warbler', '171.Myrtle_Warbler', '172.Nashville_Warbler',
                         '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler', '176.Prairie_Warbler',
                         '177.Prothonotary_Warbler', '178.Swainson_Warbler', '179.Tennessee_Warbler','180.Wilson_Warbler',
                         '181.Worm_eating_Warbler', '182.Yellow_Warbler', '183.Northern_Waterthrush','184.Louisiana_Waterthrush',
                         '185.Bohemian_Waxwing', '186.Cedar_Waxwing', '187.American_Three_toed_Woodpecker','188.Pileated_Woodpecker',
                         '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker', '191.Red_headed_Woodpecker','192.Downy_Woodpecker',
                         '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '196.House_Wren',
                         '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat')
# NETS = {'vgg16': ('VGG16',
#                   'VGG16_faster_rcnn_final.caffemodel'),
#         'zf': ('ZF',
#                   'ZF_faster_rcnn_final.caffemodel')}
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_iter_10000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
