import os

# Dataset root path
'''
    define your root directory
'''
ROOT = '/home/hbae0830/home/test/animal-re-identification/dataset'

 

# Model settings
MEGAD_NAME = 'hf-hub:BVRA/MegaDescriptor-L-384'
EVA_NAME = 'EVA02-L-14-336'
EVA_WEIGHT_NAME = 'merged2b_s6b_b61k'
DEVICE = 'cuda'

# Threshold
THRESHOLD = 0.35

# Router settings (systematic trust routing)
ROUTER_BASE_WEIGHTS = {
    "mega": 0.35,
    "aliked": 0.30,
    "eva": 0.35,
}
ROUTER_CONFIDENCE_MIX = 0.6

# Router auto tuning on calibration split
ROUTER_AUTO_TUNE = True
ROUTER_TUNE_VAL_RATIO = 0.3
ROUTER_TUNE_TOPK = 25
ROUTER_TUNE_WEIGHT_STEP = 0.1
ROUTER_TUNE_MIX_CANDIDATES = [0.4, 0.5, 0.6, 0.7, 0.8]