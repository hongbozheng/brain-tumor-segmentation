import torch
import torch.nn as nn
from logger import LogLevel
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

""" =============== Swin UNETR =============== """
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.ROI = [128, 128, 128]
_C.MODEL.SWIN.IN_CHANNELS = 4
_C.MODEL.SWIN.OUT_CHANNELS = 3
_C.MODEL.SWIN.DEPTHS = [2, 2, 2, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.FEATURE_SIZE = 48
_C.MODEL.SWIN.NORM_NAME = "instance"
_C.MODEL.SWIN.DROP_RATE = 0.0
_C.MODEL.SWIN.ATTN_DROP_RATE = 0.0
_C.MODEL.SWIN.DROPOUT_PATH_RATE = 0.0
_C.MODEL.SWIN.NORMALIZE = True
_C.MODEL.SWIN.USE_CHECKPOINT = True
_C.MODEL.SWIN.SPATIAL_DIMS = 3
_C.MODEL.SWIN.DOWNSAMPLE = "mergingv2"
_C.MODEL.SWIN.USE_V2 = True

""" AdamW """
_C.MODEL.SWIN.LR = 1e-4
_C.MODEL.SWIN.WEIGHT_DECAY = 1e-5


""" =============== UNETR =============== """
_C.MODEL.UNETR = CN()
_C.MODEL.UNETR.IN_CHANNELS = 4
_C.MODEL.UNETR.OUT_CHANNELS = 3
_C.MODEL.UNETR.ROI = [128, 128, 128]
_C.MODEL.UNETR.FEATURE_SIZE = 16
_C.MODEL.UNETR.HIDDEN_SIZE = 768
_C.MODEL.UNETR.MLP_DIM = 3072
_C.MODEL.UNETR.NUM_HEADS = 12
_C.MODEL.UNETR.PROJ_TYPE = "conv"
_C.MODEL.UNETR.NORM_NAME = "instance"
_C.MODEL.UNETR.CONV_BLOCK = True
_C.MODEL.UNETR.RES_BLOCK = True
_C.MODEL.UNETR.DROPOUT_RATE = 0.0
_C.MODEL.UNETR.SPATIAL_DIMS = 3
_C.MODEL.UNETR.QKV_BIAS = False
_C.MODEL.UNETR.SAVE_ATTN = False

""" AdamW """
_C.MODEL.UNETR.LR = 1e-4
_C.MODEL.UNETR.WEIGHT_DECAY = 1e-5


""" =============== UNETRPP =============== """
_C.MODEL.UNETRPP = CN()
_C.MODEL.UNETRPP.IN_CHANNELS = 4
_C.MODEL.UNETRPP.OUT_CHANNELS = 3
_C.MODEL.UNETRPP.ROI = [128, 128, 128]
_C.MODEL.UNETRPP.FEATURE_SIZE = 16
_C.MODEL.UNETRPP.HIDDEN_SIZE = 256
_C.MODEL.UNETRPP.NUM_HEADS = 4
_C.MODEL.UNETRPP.POS_EMBED = "perceptron"
_C.MODEL.UNETRPP.NORM_NAME = "instance"
_C.MODEL.UNETRPP.DROPOUT_RATE = 0.0
_C.MODEL.UNETRPP.DEPTHS = [3, 3, 3, 3]
_C.MODEL.UNETRPP.DIMS = [32, 64, 128, 256]
# _C.MODEL.UNETR_PP.CONV_OP = nn.Conv3d
_C.MODEL.UNETRPP.DO_DS = False

""" SGD """
_C.MODEL.UNETRPP.LR = 3e-4  # 1e-2
_C.MODEL.UNETRPP.MOMENTUM = 3e-5
_C.MODEL.UNETRPP.WEIGHT_DECAY = 0.99
_C.MODEL.UNETRPP.NESTEROV = True


""" =============== nnFormer =============== """
_C.MODEL.NNFORMER = CN()
_C.MODEL.NNFORMER.CROP_SIZE = [128, 128, 128]
_C.MODEL.NNFORMER.EMBEDDING_DIM = 96
_C.MODEL.NNFORMER.INPUT_CHANNELS = 4
_C.MODEL.NNFORMER.NUM_CLASSES = 3
# _C.MODEL.NNFORMER.CONV_OP = nn.Conv3d
_C.MODEL.NNFORMER.DEPTHS = [2, 2, 2, 2]
_C.MODEL.NNFORMER.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.NNFORMER.PATCH_SIZE = [4, 4, 4]
_C.MODEL.NNFORMER.WINDOW_SIZE = [4, 4, 8, 4]
_C.MODEL.NNFORMER.DEEP_SUPERVISION = False

""" SGD """
_C.MODEL.NNFORMER.LR = 3e-4  # 1e-2
_C.MODEL.NNFORMER.MOMENTUM = 3e-5
_C.MODEL.NNFORMER.WEIGHT_DECAY = 0.99
_C.MODEL.NNFORMER.NESTEROV = True


""" =============== UNet3D =============== """
_C.MODEL.UNET3D = CN()
_C.MODEL.UNET3D.IN_CHANNELS = 4
_C.MODEL.UNET3D.ROI = [128, 128, 128]
_C.MODEL.UNET3D.KERNELS = [
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3]
]
_C.MODEL.UNET3D.STRIDES = [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2]
]

""" Adam """
_C.MODEL.UNET3D.LR = 5e-4  # 7e-4 9e-4
_C.MODEL.UNET3D.WEIGHT_DECAY = 1e-4


# -----------------------------------------------------------------------------
# Hyperparams
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
LOG_LEVEL = LogLevel.info


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

""" Training """
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NUM_WORKERS = 2
_C.TRAIN.PIN_MEMORY = True
_C.TRAIN.WARMUP_EPOCHS = 15
_C.TRAIN.N_EPOCHS = 100


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()

""" Validation """
_C.VAL.SW_BATCH_SIZE = 2
_C.VAL.OVERLAP = 0.5


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
_C.SAVE = CN()

""" Model """
_C.SAVE.MODEL_DIR = "models"
_C.SAVE.SWIN_BEST_MODEL = _C.SAVE.MODEL_DIR + "/swin_unetr_best.ckpt"
_C.SAVE.UNETR_BEST_MODEL = _C.SAVE.MODEL_DIR + "/unetr_best.ckpt"
_C.SAVE.UNETRPP_BEST_MODEL = _C.SAVE.MODEL_DIR + "/unetrpp_best.ckpt"
_C.SAVE.NNFORMER_BEST_MODEL = _C.SAVE.MODEL_DIR + "/nnformer_best.ckpt"
_C.SAVE.UNET3D_BEST_MODEL = _C.SAVE.MODEL_DIR + "/unet3d_best.ckpt"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()

""" BraTS 2023 """
_C.DATA.DATA_DIR = "../BraTS_2023"
_C.DATA.VAL_PCT = 0.2


def get_config(args):
    """
    Get a yacs CfgNode object with default values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config