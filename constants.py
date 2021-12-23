import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG_FILE = './configs/video_narr.yml'
GPU_ID=2
DATASET='video-narr'
MODEL = 'resnet101'
QUESTION_TYPE = 'none'
VIDEO_DIR = './raw_data/test'
NUM_CLIPS = 8
BATCH_SIZE=1
IMAGE_HEIGHT=224
IMAGE_WIDTH=224
DEMO_VIDEO_PATH = './raw_data/test/원천데이터/생활안전/대본O'
DEMO_QUES_PATH = './raw_data/test/라벨링데이터/생활안전/대본O/output.json'
SAVE_DIR = './results/expVIDEO-NARR'