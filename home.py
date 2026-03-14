import streamlit as st

st.set_page_config(
    page_title="머신러닝 시각화 도구",
    layout="centered"
)

st.title("머신러닝 시각화 도구")
st.caption("왼쪽 사이드바에서 알고리즘 페이지를 선택하세요.")

st.markdown(
    """
### 구성
- KNN
- 선형회귀
- K-means

현재는 **KNN 페이지**가 먼저 구현되어 있습니다.
"""
)
