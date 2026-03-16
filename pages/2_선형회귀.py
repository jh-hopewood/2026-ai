import streamlit as st
import numpy as np
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="선형회귀",
    layout="centered"
)

# ============================================================
# 상태 초기화
# ============================================================
def reset_lr_state():
    st.session_state.lr_w = -0.5
    st.session_state.lr_b = 6.0
    st.session_state.lr_step = 0


if "lr_initialized" not in st.session_state:
    reset_lr_state()
    st.session_state.lr_initialized = True

# ============================================================
# 데이터
# ============================================================
x_data = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y_data = np.array([2.2, 2.8, 3.7, 4.1, 5.3, 5.8], dtype=float)

# ============================================================
# Helper
# ============================================================
def predict(x, w, b):
    return w * x + b


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent_step(x, y, w, b, lr):
    n = len(x)
    y_pred = predict(x, w, b)
    error = y_pred - y

    dw = (2 / n) * np.sum(error * x)
    db = (2 / n) * np.sum(error)

    new_w = w - lr * dw
    new_b = b - lr * db

    return new_w, new_b, dw, db


def build_lr_option():
    w = st.session_state.lr_w
    b = st.session_state.lr_b

    y_pred_points = predict(x_data, w, b)
    current_mse = mse(y_data, y_pred_points)

    # 데이터 점
    scatter_points = [[float(x), float(y)] for x, y in zip(x_data, y_data)]

    # 현재 직선
    x_line = np.linspace(0, 7, 100)
    y_line = predict(x_line, w, b)
    line_points = [[float(x), float(y)] for x, y in zip(x_line, y_line)]

    series = [
        {
            "name": "데이터",
            "type": "scatter",
            "data": scatter_points,
            "symbolSize": 14,
            "itemStyle": {"color": "#3498db"},
            "z": 3,
        },
        {
            "name": "현재 직선",
            "type": "line",
            "data": line_points,
            "symbol": "none",
            "lineStyle": {
                "color": "#e74c3c",
                "width": 2
            },
            "z": 2,
        }
    ]

    # 오차선 + 오차 텍스트
    for x, y_true, y_pred in zip(x_data, y_data, y_pred_points):
        # 세로 오차선
        series.append(
            {
                "name": "",
                "type": "line",
                "data": [
                    [float(x), float(y_true)],
                    [float(x), float(y_pred)]
                ],
                "symbol": "none",
                "lineStyle": {
                    "color": "#888",
                    "width": 1,
                    "type": "dashed"
                },
                "label": {"show": False},
                "tooltip": {"show": False},
                "z": 1,
            }
        )

        # 오차값 텍스트
        mid_y = float((y_true + y_pred) / 2)
        err = float(y_true - y_pred)

        series.append(
            {
                "name": "",
                "type": "scatter",
                "data": [
                    {
                        "value": [float(x), mid_y],
                        "label": {
                            "show": True,
                            "formatter": f"{err:.2f}",
                            "position": "right",
                            "fontSize": 11,
                            "color": "#333"
                        }
                    }
                ],
                "symbolSize": 1,
                "itemStyle": {"color": "rgba(0,0,0,0)"},
                "tooltip": {"show": False},
                "z": 5,
            }
        )

    option = {
        "animation": False,
        "color": ["#3498db", "#e74c3c"],
        "tooltip": {"trigger": "item"},
        "legend": {
            "top": 10,
            "data": ["데이터", "현재 직선"]
        },
        "grid": {
            "left": 55,
            "right": 30,
            "top": 55,
            "bottom": 55
        },
        "xAxis": {
            "type": "value",
            "min": 0,
            "max": 7,
            "name": "X",
            "nameLocation": "middle",
            "nameGap": 28
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 8,
            "name": "Y",
            "nameLocation": "middle",
            "nameGap": 35
        },
        "series": series,
    }

    return option, current_mse


# ============================================================
# 화면
# ============================================================
st.title("머신러닝 시각화 도구")
st.subheader("선형회귀")
st.caption("직선이 오차를 줄이는 방향으로 어떻게 수정되는지 단계별로 확인합니다.")

@st.fragment
def lr_panel():
    with st.form("lr_form"):
        col1, col2 = st.columns([1.5, 1])

        with col1:
            lr = st.slider(
                "학습률",
                min_value=0.001,
                max_value=0.1,
                value=0.03,
                step=0.001,
                key="lr_rate"
            )

        with col2:
            st.write("")
            st.write("")

        b1, b2, b3 = st.columns(3)
        with b1:
            next_clicked = st.form_submit_button("다음 단계", use_container_width=True)
        with b2:
            auto_clicked = st.form_submit_button("10단계 자동 학습", use_container_width=True)
        with b3:
            reset_clicked = st.form_submit_button("초기화", use_container_width=True)

    if next_clicked:
        new_w, new_b, _, _ = gradient_descent_step(
            x_data,
            y_data,
            st.session_state.lr_w,
            st.session_state.lr_b,
            lr
        )
        st.session_state.lr_w = new_w
        st.session_state.lr_b = new_b
        st.session_state.lr_step += 1

    if auto_clicked:
        for _ in range(10):
            new_w, new_b, _, _ = gradient_descent_step(
                x_data,
                y_data,
                st.session_state.lr_w,
                st.session_state.lr_b,
                lr
            )
            st.session_state.lr_w = new_w
            st.session_state.lr_b = new_b
            st.session_state.lr_step += 1

    if reset_clicked:
        reset_lr_state()

    option, current_mse = build_lr_option()
    st_echarts(options=option, height="560px")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.write(f"단계: {st.session_state.lr_step}")
    with col_b:
        st.write(f"기울기 w: {st.session_state.lr_w:.4f}")
    with col_c:
        st.write(f"절편 b: {st.session_state.lr_b:.4f}")
    with col_d:
        st.write(f"MSE: {current_mse:.4f}")

lr_panel()
