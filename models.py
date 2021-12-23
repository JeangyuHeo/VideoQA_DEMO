import os
import json
import torch
import torch.nn as nn
import torchvision
import streamlit as st
import model.HCRN as HCRN
from constants import device, SAVE_DIR
from preprocess.modals import resnext

@st.cache
def build_resnet():
    cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model

@st.cache
def build_resnext():
    model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
                              sample_size=112, sample_duration=16,
                              last_fc=False)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists('data/preprocess/pretrained/resnext-101-kinetics.pth')
    model_data = torch.load('data/preprocess/pretrained/resnext-101-kinetics.pth', map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    
    return model

@st.cache
def load_model():
    with open('./data/video-narr/video-narr_vocab.json','r') as f:
        vocab = json.load(f)
    
    ckpt = os.path.join(SAVE_DIR, 'ckpt', 'model.pt')
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']
    model_kwargs.update({'vocab': vocab})

    model = HCRN.HCRNNetwork(**model_kwargs).to(device)

    new_state_dict = {}
    for k, v in loaded['state_dict'].items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    
    return model