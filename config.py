import torch
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
_C.MODEL.UNETR_PP = CN()
_C.MODEL.UNETR_PP.IN_CHANNELS = 4
_C.MODEL.UNETR_PP.OUT_CHANNELS = 3
_C.MODEL.UNETR_PP.ROI = [128, 128, 128]
_C.MODEL.UNETR_PP.FEATURE_SIZE = 16
_C.MODEL.UNETR_PP.HIDDEN_SIZE = 256
_C.MODEL.UNETR_PP.NUM_HEADS = 4
_C.MODEL.UNETR_PP.POS_EMBED = "perceptron"
_C.MODEL.UNETR_PP.NORM_NAME = "instance"
_C.MODEL.UNETR_PP.DROPOUT_RATE = 0.0
_C.MODEL.UNETR_PP.DEPTHS = [3, 3, 3, 3]
_C.MODEL.UNETR_PP.DIMS = [32, 64, 128, 256]
# _C.MODEL.UNETR_PP.CONV_OP = nn.Conv3d
_C.MODEL.UNETR_PP.DO_DS = False

""" SGD """
_C.MODEL.UNETR_PP.LR = 3e-4  # 1e-2
_C.MODEL.UNETR_PP.MOMENTUM = 3e-5
_C.MODEL.UNETR_PP.WEIGHT_DECAY = 0.99
_C.MODEL.UNETR_PP.NESTEROV = True


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
_C.MODEL.UNET3D.N_CLASS = 3
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
_C.MODEL.UNET3D.NORM = "instancenorm3d"
_C.MODEL.UNET3D.DIM = 3
_C.MODEL.UNET3D.DEEP_SUPERVISION = False

""" Adam """
_C.MODEL.UNET3D.LR = 5e-4  # 7e-4 9e-4
_C.MODEL.UNET3D.WEIGHT_DECAY = 1e-4


# -----------------------------------------------------------------------------
# Best Model
# -----------------------------------------------------------------------------
_C.BEST_MODEL = CN()

""" Model """
_C.BEST_MODEL.DIR = "models"
_C.BEST_MODEL.SWIN = _C.BEST_MODEL.DIR + "/swin_unetr_best.ckpt"
_C.BEST_MODEL.UNETR = _C.BEST_MODEL.DIR + "/unetr_best.ckpt"
_C.BEST_MODEL.UNETR_PP = _C.BEST_MODEL.DIR + "/unetr_pp_best.ckpt"
_C.BEST_MODEL.NNFORMER = _C.BEST_MODEL.DIR + "/nnformer_best.ckpt"
_C.BEST_MODEL.UNET3D = _C.BEST_MODEL.DIR + "/unet3d_best.ckpt"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()

""" BraTS 2023 """
_C.DATA.DIR = "../BraTS_2023"
_C.DATA.VAL_PCT = 0.2
_C.DATA.ROI = [128, 128, 128]


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
_C.LOADER = CN()

""" DataLoader """
_C.LOADER.NUM_WORKERS = 2
_C.LOADER.PIN_MEMORY = True


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
MODEL_NAMES = {"swin", "unetr", "unetr_pp", "nnformer", "unet3d"}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

""" Training """
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.N_EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 15
_C.TRAIN.WARMUP_START_LR = 1e-5
_C.TRAIN.ETA_MIN = 1e-7
_C.TRAIN.STATS_FILEPATH = "stats.json"


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()

""" Validation """
_C.VAL.BATCH_SIZE = 1
_C.VAL.SW_BATCH_SIZE = 2
_C.VAL.OVERLAP = 0.5


def get_config(args):
    """
    Get a yacs CfgNode object with default values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config