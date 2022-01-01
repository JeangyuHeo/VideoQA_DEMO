# Video Question Answering(동영상 질의 응답)을 위한 Hierarchical Conditional Relation Networks (HCRN) 데모
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 1.9.0](https://img.shields.io/badge/pytorch-1.9.0-green.svg?style=plastic)

본 레포는 [Surromind-AI/Videonarrative](https://github.com/Surromind-AI/videonarrative)의 비공식 Inference Pipeline 데모를 제공합니다.

네이버 커넥트재단의 교육과정 [Boostcamp AI-Tech](https://boostcamp.connect.or.kr/program_ai.html) 2기 KiYOUNG2 팀의 최종 프로젝트를 위한 VideoQA 데모로 활용하기 위해 작성되었습니다.

<br>
<br>

## Model Overview
![image](https://user-images.githubusercontent.com/88299729/147853790-4b86eae7-0f42-40f0-bb03-743cd81745a9.png)

<br>
<br>

### Requirements

```
beautifulsoup4==4.6.0
cached-property==1.5.2
certifi==2021.5.30
charset-normalizer==2.0.4
click==8.0.1
colorama==0.4.4
cycler==0.10.0
easydict==1.9
h5py==3.1.0
idna==3.2
importlib-metadata==4.6.1
joblib==1.0.1
JPype1==1.3.0
kiwisolver==1.3.1
konlpy==0.5.2
lxml==4.6.3
matplotlib==3.3.4
mecab-python===0.996-ko-0.9.2
nltk==3.6.2
numpy==1.19.5
oauthlib==3.1.1
pandas==1.1.5
Pillow==8.3.1
protobuf==3.18.0
pyparsing==2.4.7
PySocks==1.7.1
python-dateutil==2.8.1
pytz==2021.1
PyYAML==5.4.1
regex==2021.7.6
requests==2.26.0
requests-oauthlib==1.3.0
scikit-video==1.1.11
scipy==1.2.0
six==1.16.0
tensorboardX==2.4
termcolor==1.1.0
torch==1.2.0
torchvision==0.2.1
tqdm==4.61.2
tweepy==3.10.0
typing-extensions==3.10.0.0
urllib3==1.26.6
zipp==3.5.0
```
<br>
<br>

### 데모를 위한 데이터 세팅
```
bash setup.py
```

### 실행 커맨드

```bash
streamlit run app.py
```

<br>
<br>
