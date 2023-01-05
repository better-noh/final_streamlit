import os
import sys
import pandas as pd
import numpy as np 
import requests
import webbrowser

# from io import BytesIO
from glob import glob
from PIL import Image, ImageEnhance

import streamlit as st

st.set_page_config(
    page_title="Online class monitoring using Face Recognition",
    page_icon="📸",
    layout="wide",
)


st.image("img/logo.png")
st.sidebar.subheader("수강생 자리비움 탐지 Model ✅")


st.sidebar.markdown("📸줌인줌아웃📸")
st.sidebar.caption("🔥노나은: 방향성 길잡이")
st.sidebar.caption("🔥문병욱: 폼 좋은 병욱")
st.sidebar.caption("🔥문영운: 갓영운")
st.sidebar.caption("🔥박혜민: 혬또잘")
st.sidebar.caption("🔥조혜인: 혠 조")
# option = st.sidebar.selectbox('Select Page',
#                     ('Home Page', 'Meiapipe', 'OpenCV', 'Face Recognition', 'Yolov3', 'Yolov5', 'Yolov7', 'Demonstration'))
# st.write('You selected:', option)
# st.sidebar.markdown("---")

# st.header("코로나19와 비대면 원격 수업")
# st.write("신종 코로나바이러스 감염증(코로나19)으로 비대면 원격 서비스 활용이 늘어나면서, <멋쟁이사자처럼 AIS> 7기 수업 역시 줌(Zoom)을 통한 온라인 수업으로 진행되고 있습니다.")
# st.write("이런 온라인 강의는 대부분 관리 효율성을 위해 어느 정도 모니터링이 필요하며, 7기 수강생들 역시 모니터링 매니저님 덕분에 수업 집중도를 높일 수 있었습니다.")
# st.write("하지만 모니터링 매니저님 혼자서 다수의 줌 화면을 확인하는 데는 피로도가 높을 것으로 판단하였고, 저희 팀은 모니터링 매니저님의 업무 효율성을 높일 수 있는, <온라인 화상 수업 서포팅 프로그램> 을 프로젝트 주제로 선정했습니다.")
# st.image("img/주제_선정_배경_resize.png")


# =======
#   App
# =======

# provide options to either select an image form the Topic, model one, or demonstration
topic_tab, model_tab, demon_tab = st.tabs(["Topic", "Model", "Demonstration"])
with topic_tab:
    st.subheader("코로나19와 비대면 원격 수업")
    st.write("신종 코로나바이러스 감염증(코로나19)으로 비대면 원격 서비스 활용이 늘어나면서, <멋쟁이사자처럼 AIS> 7기 수업 역시 줌(Zoom)을 통한 온라인 수업으로 진행되고 있습니다.")
    st.write("이런 온라인 강의는 대부분 관리 효율성을 위해 어느 정도 모니터링이 필요하며, 7기 수강생들 역시 모니터링 매니저님 덕분에 수업 집중도를 높일 수 있었습니다.")
    st.write("하지만 모니터링 매니저님 혼자서 다수의 줌 화면을 확인하는 데는 피로도가 높을 것으로 판단하였고, 저희 팀은 모니터링 매니저님의 업무 효율성을 높일 수 있는, <온라인 화상 수업 서포팅 프로그램> 을 프로젝트 주제로 선정했습니다.")
    st.image("img/주제_선정_배경_resize.png")

with model_tab:
    option = st.selectbox('Select Page',
                        ('Please select Model', 'Mediapipe', 'OpenCV', 'Face Recognition', 'Yolov3', 'Yolov5', 'Yolov7'))
    # st.write('You selected:', option)
    # tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs('Mediapipe', 'OpenCV', 'Face Recognition', 'Yolov3', 'Yolov5', 'Yolov7')
    # st.write('You selected:', option)
    
    st.header("Mediapipe")
    st.write(
                f"""
                    - 구글에서 주로 인체를 대상으로 하는 비전인식기능들을 AI모델 개발과 기계 학습까지 마친 상태로 제공하는 서비스
                    - 다양한 프로그램 언어에서 사용하기 편하게 라이브러리 형태로 모듈화되어 제공되며 사용 방법 또한 풍부하게 제공되기 때문에 몇 가지 간단한 단계로 Mediapipe에서 제공하는 AI기능을 활용한 응용 프로그램 개발이 가능
                    """
            )
    st.header("MediaPipe의 Face Detection")
    st.write(
                f"""
                    - 6개의 랜드마크(오른쪽 눈, 왼쪽 눈, 코 끝, 입 중심, 오른쪽 귀 윗 가장자리 위의 점 및 왼쪽 귀 윗 가장자리 위의 점) 및 다중 얼굴 지원과 함께 제공되는 초고속 얼굴 감지 솔루션
                    """
            )
    
    st.header("OpenCV")
    st.write(f"""
                - Open Source Computer Vision의 약자로, 영상 처리에 사용할 수 있는 오픈 소스 라이브러리
                - 컴퓨터가 사람의 눈처럼 인식할 수 있게 처리해주는 역할을 하기도 하며, 카메라 어플에서도 OpenCV가 사용됨
                - 추가로 사용되는 예시 : 공장에서 제품 검사, 의료 영상 처리 및 보정 그리고 판단,
    CCTV영상, 로보틱스
             """)
    
    st.header("Face Recognition")
    st.write(f"""
                - 해당 라이브러리는 딥러닝 기반으로 제작된 [dlib](http://dlib.net/)의 얼굴 인식 기능을 사용하여 구축
                - 이 모델은 [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) 기반으로 99.38%의 정확도를 가짐([https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition))
             """)

with demon_tab:
    st.subheader("자리비움 탐지 시연 영상")
    video_file = open('img/final.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
    url = "https://www.youtube.com/watch?v=oMhG6HIHy_g"
    if st.button("Open YouTube"):
        webbrowser.open_new_tab(url)

# st.sidebar.success(print_praise())   
st.sidebar.write("---\n")
st.sidebar.caption("""You can check out the source code [here](https://github.com/better-noh/final_streamlit).""")
st.sidebar.write("---\n")