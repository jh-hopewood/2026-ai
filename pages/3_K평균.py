import streamlit as st
import numpy as np
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="K-means",
    layout="centered"
)

# ============================================================
# 상태 초기화
# ============================================================
def reset_kmeans_state():
    st.session_state.km_centroids = None
    st.session_state.km_labels = None
    st.session_state.km_iteration = 0
    st.session_state.km_phase = "init"   # init -> assign -> update
    st.session_state.km_converged = False
    st.session_state.km_paths = None


if "km_initialized" not in st.session_state:
    reset_kmeans_state()
    st.session_state.km_initialized = True

# ============================================================
# 데이터
# ============================================================
X = np.array([
    [0.5, 5.0], [1.0, 5.8], [1.2, 6.3], [1.6, 5.4],
    [2.0, 4.9], [2.3, 5.9], [2.6, 4.8], [2.9, 5.3],
    [6.0, 4.5], [6.5, 3.8], [7.0, 3.0], [7.5, 2.5],
    [8.0, 2.2], [8.5, 3.0], [9.0, 2.6], [7.8, 3.8],
    [6.2, 7.0], [6.8, 8.0], [7.2, 8.5], [7.8, 7.2],
    [8.2, 8.0], [8.8, 7.5], [9.2, 8.3], [7.5, 6.8]
], dtype=float)

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

# ============================================================
# Helper
# ============================================================
def initialize_centroids(k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx].copy()


def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, centroids):
    old_centroids = centroids.copy()
    new_centroids = []

    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(centroids[i])

    new_centroids = np.array(new_centroids)
    movement = np.linalg.norm(new_centroids - old_centroids, axis=1)
    converged = np.all(movement < 1e-4)

    return new_centroids, converged


def inertia(X, labels, centroids):
    total = 0.0
    for i in range(len(X)):
        total += np.sum((X[i] - centroids[labels[i]]) ** 2)
    return total


def build_kmeans_option(k):
    labels = st.session_state.km_labels
    centroids = st.session_state.km_centroids
    paths = st.session_state.km_paths

    series = []

    # 아직 군집 배정 전이면 회색 점
    if labels is None:
        gray_points = []
        for i, point in enumerate(X):
            gray_points.append(
                {
                    "value": [float(point[0]), float(point[1])],
                    "label": {
                        "show": True,
                        "formatter": str(i),
                        "position": "right",
                        "fontSize": 10,
                        "color": "#333"
                    }
                }
            )

        series.append(
            {
                "name": "데이터",
                "type": "scatter",
                "data": gray_points,
                "symbolSize": 12,
                "itemStyle": {"color": "#7f8c8d"},
                "z": 2,
            }
        )
    else:
        for cluster_id in range(len(centroids)):
            cluster_points = []
            idxs = np.where(labels == cluster_id)[0]

            for i in idxs:
                point = X[i]
                cluster_points.append(
                    {
                        "value": [float(point[0]), float(point[1])],
                        "label": {
                            "show": True,
                            "formatter": str(i),
                            "position": "right",
                            "fontSize": 10,
                            "color": "#333"
                        }
                    }
                )

            series.append(
                {
                    "name": f"군집 {cluster_id + 1}",
                    "type": "scatter",
                    "data": cluster_points,
                    "symbolSize": 12,
                    "itemStyle": {"color": COLORS[cluster_id]},
                    "z": 2,
                }
            )

    # 중심점 이동 경로
    if paths is not None:
        for i, path in enumerate(paths):
            if len(path) >= 2:
                line_points = [[float(p[0]), float(p[1])] for p in path]
                series.append(
                    {
                        "name": "",
                        "type": "line",
                        "data": line_points,
                        "symbol": "none",
                        "lineStyle": {
                            "color": COLORS[i],
                            "width": 1.5,
                            "type": "dashed"
                        },
                        "label": {"show": False},
                        "tooltip": {"show": False},
                        "z": 1,
                    }
                )

            # 경로상의 점들
            path_points = [[float(p[0]), float(p[1])] for p in path]
            series.append(
                {
                    "name": "",
                    "type": "scatter",
                    "data": path_points,
                    "symbolSize": 8,
                    "itemStyle": {"color": COLORS[i]},
                    "label": {"show": False},
                    "tooltip": {"show": False},
                    "z": 3,
                }
            )

    # 현재 중심점
    if centroids is not None:
        for i, c in enumerate(centroids):
            series.append(
                {
                    "name": f"중심 {i + 1}",
                    "type": "scatter",
                    "data": [
                        {
                            "value": [float(c[0]), float(c[1])],
                            "label": {
                                "show": True,
                                "formatter": f"C{i+1}",
                                "position": "top",
                                "fontSize": 12,
                                "color": "#111"
                            }
                        }
                    ],
                    "symbol": "diamond",
                    "symbolSize": 20,
                    "itemStyle": {
                        "color": COLORS[i],
                        "borderColor": "#111",
                        "borderWidth": 1.5
                    },
                    "z": 5,
                }
            )

    option = {
        "animation": False,
        "tooltip": {"trigger": "item"},
        "legend": {
            "top": 10
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

    return option

# ============================================================
# 화면
# ============================================================
st.title("머신러닝 시각화 도구")
st.subheader("K 평균(K-means)")
st.caption("중심점이 이동하며 군집을 형성하는 과정을 단계별로 확인합니다.")

@st.fragment
def kmeans_panel():
    with st.form("kmeans_form"):
        col1, col2 = st.columns([1.5, 1])

        with col1:
            k_val = st.slider(
                "K값",
                min_value=2,
                max_value=5,
                value=3,
                key="km_k"
            )

        with col2:
            st.write("")
            st.write("")

        b1, b2, b3 = st.columns(3)
        with b1:
            init_clicked = st.form_submit_button("초기 중심 배치", use_container_width=True)
        with b2:
            next_clicked = st.form_submit_button("다음 단계", use_container_width=True)
        with b3:
            reset_clicked = st.form_submit_button("초기화", use_container_width=True)

    if init_clicked:
        reset_kmeans_state()
        centroids = initialize_centroids(k_val)
        st.session_state.km_centroids = centroids
        st.session_state.km_paths = [[tuple(c)] for c in centroids]
        st.session_state.km_phase = "assign"

    if next_clicked:
        if st.session_state.km_centroids is not None and not st.session_state.km_converged:
            if st.session_state.km_phase == "assign":
                st.session_state.km_labels = assign_clusters(X, st.session_state.km_centroids)
                st.session_state.km_phase = "update"

            elif st.session_state.km_phase == "update":
                new_centroids, converged = update_centroids(
                    X,
                    st.session_state.km_labels,
                    st.session_state.km_centroids
                )
                st.session_state.km_centroids = new_centroids
                st.session_state.km_iteration += 1
                st.session_state.km_converged = converged

                for i, c in enumerate(new_centroids):
                    st.session_state.km_paths[i].append(tuple(c))

                if not converged:
                    st.session_state.km_phase = "assign"

    if reset_clicked:
        reset_kmeans_state()

    option = build_kmeans_option(k_val)
    st_echarts(options=option, height="560px")

    if st.session_state.km_centroids is None:
        st.info("`초기 중심 배치`를 눌러 시작하세요.")
    else:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.write(f"반복 횟수: {st.session_state.km_iteration}")

        with col_b:
            if st.session_state.km_converged:
                st.write("현재 단계: 수렴 완료")
            elif st.session_state.km_phase == "assign":
                st.write("현재 단계: 배정 단계")
            else:
                st.write("현재 단계: 중심 이동 단계")

        with col_c:
            if st.session_state.km_labels is not None:
                current_inertia = inertia(
                    X,
                    st.session_state.km_labels,
                    st.session_state.km_centroids
                )
                st.write(f"거리 제곱합: {current_inertia:.3f}")

        if st.session_state.km_converged:
            st.success("수렴이 완료되었습니다. 더 이상 중심점이 거의 움직이지 않습니다.")

kmeans_panel()
