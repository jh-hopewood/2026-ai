import streamlit as st
import numpy as np
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="KNN",
    layout="centered"
)

# ============================================================
# 상태 초기화
# ============================================================
def reset_knn_state():
    st.session_state.knn_new_point = None
    st.session_state.knn_step = 0
    st.session_state.knn_distances = None
    st.session_state.knn_sorted_idx = None
    st.session_state.knn_final_label = None


if "knn_initialized" not in st.session_state:
    reset_knn_state()
    st.session_state.knn_initialized = True

# ============================================================
# 데이터
# ============================================================
X_knn = np.array([
    [1, 7], [2, 8], [3, 7], [4, 6],
    [7, 2], [8, 3], [9, 2], [6, 4]
], dtype=float)

y_knn = np.array([
    "빨강", "빨강", "빨강", "빨강",
    "파랑", "파랑", "파랑", "파랑"
])

CLASS_COLORS = {
    "빨강": "#e74c3c",
    "파랑": "#3498db",
    "새 데이터": "#2ecc71"
}

# ============================================================
# KNN helper
# ============================================================
def predict_knn(point, k):
    dists = np.linalg.norm(X_knn - point, axis=1)
    idx = np.argsort(dists)[:k]
    neighbor_labels = y_knn[idx]

    unique, counts = np.unique(neighbor_labels, return_counts=True)
    max_count = np.max(counts)
    winners = unique[counts == max_count]

    if len(winners) == 1:
        return winners[0]

    return y_knn[np.argsort(dists)[0]]


def prepare_knn(new_point, k):
    st.session_state.knn_step = 0
    st.session_state.knn_new_point = np.array(new_point, dtype=float)

    distances = np.linalg.norm(X_knn - st.session_state.knn_new_point, axis=1)
    sorted_idx = np.argsort(distances)

    st.session_state.knn_distances = distances
    st.session_state.knn_sorted_idx = sorted_idx
    st.session_state.knn_final_label = predict_knn(st.session_state.knn_new_point, k)


def build_knn_option(k):
    red_points = []
    blue_points = []

    for i, (point, label) in enumerate(zip(X_knn, y_knn)):
        item = {
            "value": [float(point[0]), float(point[1])],
            "label": {
                "show": True,
                "formatter": str(i),
                "position": "right",
                "fontSize": 11,
                "color": "#333"
            }
        }
        if label == "빨강":
            red_points.append(item)
        else:
            blue_points.append(item)

    series = [
        {
            "name": "빨강",
            "type": "scatter",
            "data": red_points,
            "symbolSize": 14,
            "itemStyle": {"color": CLASS_COLORS["빨강"]},
            "z": 3,
        },
        {
            "name": "파랑",
            "type": "scatter",
            "data": blue_points,
            "symbolSize": 14,
            "itemStyle": {"color": CLASS_COLORS["파랑"]},
            "z": 3,
        },
    ]

    new_point = st.session_state.knn_new_point
    sorted_idx = st.session_state.knn_sorted_idx
    step = st.session_state.knn_step
    distances = st.session_state.knn_distances

    if new_point is not None:
        series.append(
            {
                "name": "새 데이터",
                "type": "scatter",
                "data": [
                    {
                        "value": [float(new_point[0]), float(new_point[1])],
                        "label": {
                            "show": True,
                            "formatter": "새",
                            "position": "top",
                            "fontSize": 12,
                            "color": "#111"
                        }
                    }
                ],
                "symbol": "triangle",
                "symbolSize": 18,
                "itemStyle": {"color": CLASS_COLORS["새 데이터"]},
                "z": 5,
            }
        )

    if new_point is not None and sorted_idx is not None:
        current_neighbors = min(step, k)

        for i in range(current_neighbors):
            idx = sorted_idx[i]
            neighbor = X_knn[idx]
            label_name = y_knn[idx]

            series.append(
                {
                    "name": "",
                    "type": "line",
                    "data": [
                        [float(new_point[0]), float(new_point[1])],
                        [float(neighbor[0]), float(neighbor[1])]
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

            series.append(
                {
                    "name": "",
                    "type": "scatter",
                    "data": [[float(neighbor[0]), float(neighbor[1])]],
                    "symbolSize": 26,
                    "itemStyle": {
                        "color": "rgba(0,0,0,0)",
                        "borderColor": CLASS_COLORS[label_name],
                        "borderWidth": 2,
                    },
                    "label": {"show": False},
                    "tooltip": {"show": False},
                    "z": 4,
                }
            )

            mid_x = float((new_point[0] + neighbor[0]) / 2)
            mid_y = float((new_point[1] + neighbor[1]) / 2)

            series.append(
                {
                    "name": "",
                    "type": "scatter",
                    "data": [
                        {
                            "value": [mid_x, mid_y],
                            "label": {
                                "show": True,
                                "formatter": f"{i+1}NN\n{distances[idx]:.2f}",
                                "position": "top",
                                "fontSize": 11,
                                "color": "#333"
                            }
                        }
                    ],
                    "symbolSize": 1,
                    "itemStyle": {"color": "rgba(0,0,0,0)"},
                    "tooltip": {"show": False},
                    "z": 6,
                }
            )

    return {
        "animation": False,
        "color": [
            CLASS_COLORS["빨강"],
            CLASS_COLORS["파랑"],
            CLASS_COLORS["새 데이터"]
        ],
        "tooltip": {"trigger": "item"},
        "legend": {
            "top": 10,
            "data": ["빨강", "파랑", "새 데이터"]
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
            "max": 10,
            "name": "X",
            "nameLocation": "middle",
            "nameGap": 28
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 10,
            "name": "Y",
            "nameLocation": "middle",
            "nameGap": 35
        },
        "series": series,
    }

# ============================================================
# 화면
# ============================================================
st.title("머신러닝 시각화 도구")
st.subheader("K 최근접 이웃(KNN)")
st.caption("좌표를 입력한 뒤 점 찍기 버튼으로 새 데이터를 추가합니다.")

@st.fragment
def knn_panel():
    with st.form("knn_input_form"):
        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            x_val = st.number_input(
                "X 좌표",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                key="knn_x"
            )

        with col2:
            y_val = st.number_input(
                "Y 좌표",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                key="knn_y"
            )

        with col3:
            k_val = st.slider(
                "K값",
                min_value=1,
                max_value=7,
                value=3,
                key="knn_k"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            plot_clicked = st.form_submit_button("점 찍기", use_container_width=True)
        with c2:
            next_clicked = st.form_submit_button("다음 단계", use_container_width=True)
        with c3:
            reset_clicked = st.form_submit_button("초기화", use_container_width=True)

    if plot_clicked:
        prepare_knn([x_val, y_val], k_val)

    if next_clicked:
        if st.session_state.knn_new_point is not None and st.session_state.knn_step < k_val:
            st.session_state.knn_step += 1

    if reset_clicked:
        reset_knn_state()

    if st.session_state.knn_new_point is None:
        st.info("좌표를 입력하고 `점 찍기`를 누르세요.")

    else:
        x, y = st.session_state.knn_new_point

        if st.session_state.knn_step < k_val:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.write(f"선택된 좌표: ({x:.2f}, {y:.2f})")
            with col_b:
                st.write(f"현재 단계: {st.session_state.knn_step} / {k_val}")
        else:
            neighbor_labels = y_knn[st.session_state.knn_sorted_idx[:k_val]]
            red_count = int(np.sum(neighbor_labels == "빨강"))
            blue_count = int(np.sum(neighbor_labels == "파랑"))

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write(f"선택된 좌표: ({x:.2f}, {y:.2f})")
            with col_b:
                st.success(
                    f"최종 결과 — 빨강: {red_count}표 / 파랑: {blue_count}표 → {st.session_state.knn_final_label}"
                )
                
    option = build_knn_option(k_val)
    st_echarts(options=option, height="560px")
knn_panel()
