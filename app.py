import io
import os
import json
import random
import asyncio
import streamlit as st
from confirm_button_hack import cache_on_button_press
from video_preprocess import app_motion_features
from predict import validation
from question_preprocess import process_question_multichoice
from constants import (
    DEMO_VIDEO_PATH, 
    DEMO_QUES_PATH,
    NUM_CLIPS,
)
from models import (
    build_resnet,
    build_resnext,
    load_model,
)

#st.set_page_config(layout='wide')

def main():
    st.title("안녕하세요 ! Video Question Answering에 오신 것을 환영합니다!")
    
    #name = st.text_input("이름을 입력해주세요!","")

    model_net = build_resnet()
    model_next = build_resnext()
    model = load_model()

    with open(DEMO_QUES_PATH, 'r') as f:
        question_list = json.load(f)
    btn_solve_problem =st.button("새로운 문제 풀기")

    if btn_solve_problem:    
        asyncio.run(solve_problem(question_list, model_net, model_next, model))

async def solve_problem(question_list,model_net, model_next, model):
    question_num=random.randint(0,539)

    video_path = os.path.join(DEMO_VIDEO_PATH, question_list[question_num]['vid'])
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    st.subheader(question_list[question_num]['que'])
    for idx, ans in enumerate(question_list[question_num]['answers']):
        st.write(str(idx+1)+". "+ans)

    app_feat, motion_feat = app_motion_features(video_path, NUM_CLIPS, model_net, model_next)

    q_dic = {'question':question_list[question_num]['que'], 'answers': question_list[question_num]['answers']}
    q_emb = process_question_multichoice(q_dic)

    pred = validation(q_emb,app_feat, motion_feat, model)
    
    answer = "정답은 "+ str(int(pred)+1)+ "번 입니다!!"
    st.success(answer)

main()