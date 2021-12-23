import torch
import numpy as np
from scipy.misc import imresize
from PIL import Image
import skvideo.io
import skvideo.datasets
bbb = skvideo.datasets.bigbuckbunny()
from constants import NUM_CLIPS

def extract_clips_feat(path, num_clips, width, height, num_frames_per_clip, model):
    clips = list()
    video_data = skvideo.io.vread(path.strip())
    
    total_frames = video_data.shape[0]
    img_size = (height, width)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j]
            img = Image.fromarray(frame_data)
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            frame_data = np.array(img)
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        if model in ['resnext101']:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
        
    return clips

def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats

def app_motion_features(path, num_clips, model_net, model_next):
    app_clips = extract_clips_feat(
        path=path, 
        num_clips=num_clips, 
        width=224, 
        height=224, 
        num_frames_per_clip=16,
        model='resnet101'
    )
    
    motion_clips = extract_clips_feat(
        path=path, 
        num_clips=num_clips, 
        width=112, 
        height=112,
        num_frames_per_clip=16,
        model='resnext101'
    )
    
    clip_feat=[]
    for clip_id, clip in enumerate(app_clips):
        feats = run_batch(clip, model_net)  # (16, 2048)
        feats = feats.squeeze()
        clip_feat.append(feats)
    app_feat = np.array(clip_feat)
    
    clip_torch = torch.FloatTensor(np.asarray(motion_clips)).cuda()
    motion_feat = model_next(clip_torch)  # (8, 2048)
    motion_feat = motion_feat.squeeze()
    motion_feat = motion_feat.detach().cpu().numpy()
    
    return app_feat, motion_feat