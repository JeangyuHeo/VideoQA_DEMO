import io
import torch
import numpy as np
import streamlit as st
from utils import todevice
from constants import (
    device,
    QUESTION_TYPE,
    BATCH_SIZE,
)

def validation(q_emb, app_feat, motion_feat, model):
    model.eval()
    print('validating...')
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []

    with torch.no_grad():
        answers=0
        t_question = torch.LongTensor(np.asarray(q_emb['question']))
        t_question_len = torch.LongTensor(np.asarray(q_emb['question_len']))
        t_ans_candidate = torch.LongTensor(np.asarray(q_emb['ans_candidate']))
        t_ans_candidates_len = torch.LongTensor(np.asarray(q_emb['ans_candidate_len']))
        t_appearance_feat = torch.from_numpy(np.array(app_feat)).unsqueeze(0)
        t_motion_feat = torch.from_numpy(motion_feat).unsqueeze(0)
        
        t_list = [t_ans_candidate, t_ans_candidates_len, t_appearance_feat, t_motion_feat, t_question, t_question_len]
        t_list = todevice(t_list ,device)

        try:
            logits = model(*t_list)
        except Exception as e:
            return "unanswerable"

        if QUESTION_TYPE in ['action', 'transition','none']:
            preds = torch.argmax(logits.view(BATCH_SIZE, 5), dim=1)
            agreeings = (preds == answers)
        elif QUESTION_TYPE == 'count':
            answers = answers.unsqueeze(-1)
            preds = (logits + 0.5).long().clamp(min=1, max=10)
            batch_mse = (preds - answers) ** 2
        else:
            preds = logits.detach().argmax(1)
            agreeings = (preds == answers)
    
    return preds