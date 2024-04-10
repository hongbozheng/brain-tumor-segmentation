import torch
from logger import LogLevel


""" Swin UNeTR """
ROI = (128, 128, 128)
IN_CHANNELS = 4
OUT_CHANNELS = 3
DEPTHS = (2, 2, 2, 2)
NUM_HEADS = (3, 6, 12, 24)
FEATURE_SIZE = 48
NORM_NAME = "instance"
DROP_RATE = 0.0
ATTN_DROP_RATE = 0.0
DROPOUT_PATH_RATE = 0.0
NORMALIZE = True
USE_CHECKPOINT = True
SPATIAL_DIMS = 3
DOWNSAMPLE = "merging"
USE_V2 = False


""" AdamW """
LR = 1e-4
WEIGHT_DECAY = 1e-5


""" Training """
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
VAL_PCT = 0.2
BATCH_SIZE = 1
NUM_WORKERS = 2
PIN_MEMORY = True
WARMUP_EPOCHS = 15
N_EPOCHS = 100
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
LOG_LEVEL = LogLevel.info


""" Validation """
SW_BATCH_SIZE = 2
OVERLAP = 0.5


""" Model """
MODEL_DIR = "models"
SWIN_UNETR_BEST_MODEL = MODEL_DIR + "/swin_unetr_best.ckpt"


""" BraTS 2023 """
DATA_DIR = "../BraTS_2023"