import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from llm import get_ai_response


st.set_page_config(page_title="한양대 생활관 챗봇", page_icon="🦁")


col1, col2 = st.columns([4, 1])
with col1:
    st.title("한양대 학생생활관 챗봇-Beta")
    
with col2:
    st.image("2023하이리온_초롱초롱 (1).svg", width=100) # 이미지 크기 조절

st.info("2025-2학기 내국인 정규입사 입사 등록 안내문, Foreign Student Guidebook을 기반으로 답해드립니다!\n\n*이외 내용 문의 시, 할루시네이션이 발생할 수 있습니다.")


user_selection = st.segmented_control(label='[  T/O를 선택하세요! 미선택 시, 내국인 내용으로 안내됩니다.  ]', options=['내국인(Domestic)', '외국인(Foreign)'], selection_mode="single")
if user_selection=='외국인(Foreign)':
    filter = {'student':'foreign'}
else:
    filter = {'student':'domestic'}


load_dotenv()


if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])




if user_question := st.chat_input(placeholder="생활관 입사에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question, filter, model='gemini-2.5-flash')
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})