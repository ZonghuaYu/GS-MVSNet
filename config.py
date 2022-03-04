import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import warnings
warnings.filterwarnings('ignore')

"""
 logging info format 
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s: %(message)s")

""" 
device args 
"""
import random,numpy,torch
# fix random,set device
seed_id = 1
random.seed(seed_id)
numpy.random.seed(seed_id)
torch.manual_seed(seed_id)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(seed_id)
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

logging.info("current device:"+str(DEVICE))


""" 
net args
"""
from net import core
from net.unit import scale, backbone, depthhypos, homoaggregate, regular, regress

# Number of depth assumption planes
ndepths = [48, 8, 8, 8, 8]
# homo warping groups
ngroups = 8
# scale img & matrix method
scalmodule = scale.scale_imgcam
# Feature map extraction network
featuremodule = backbone.FPN_2layers(inner_channels=[3, 16, 32, 64])
# Depth hypothesis method
depthhyposmodule = depthhypos.Depthhypos()
# Cost volume construction and aggregation method
homomodule = homoaggregate.homo_aggregate_by_variance
# 3D convolution regularization method
regularmodule = regular.RegularNet()
# Depth and confidence regression method
regressmodule = [regress.depth_regression, regress.confidence_regress]
model = core.GS_MVSNet(scalmodule, featuremodule, depthhyposmodule, homomodule,
                       regularmodule, regressmodule, ndepths, ngroups)
logging.info('>>>Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


""" 
trian args
"""
start_epoch = 1
lr = 0.001
factor = 0.9
train_nviews = 3
# pth file save path
pth_path='pth'
os.makedirs(pth_path, exist_ok=True)

"""
eval args
"""
eval_output_path = "outputs"

""" 
dtu dataset args 
"""
root_dir = os.path.join("/data", "user10")

##>>>> train
train_root_dir= os.path.join(root_dir, "dtu-train-128")
train_pair_path = os.path.join(train_root_dir,"Cameras","pair.txt")

train_label = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
              45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
              74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
              101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
              121, 122, 123, 124, 125, 126, 127, 128]
train_lighting_label = [0, 1, 2, 3, 4, 5, 6]

##>>>> val
val_nviews = 5
val_label = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
val_lighting_label = [3]

##>>>> eval
eval_root_dir = os.path.join(root_dir, "dtu")
eval_pair_path = os.path.join(eval_root_dir,"pair.txt")
eval_label = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]

""" 
tanks dataset args 
"""
datasetpath = os.path.join(root_dir, "TankandTemples", "intermediate")
scenelist = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']


if __name__=="__main__":
    pass



