import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
FEATURES = ["로그 스크린 수", "로그 누적 관객", "10위권 일수", "롱런 지수"]
CLUSTER_LABELS = ["㉮", "㉯", "㉰"]

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.write("영화의 흥행 규모와 롱런 특성을 바탕으로 비슷한 영화끼리 묶어 봅니다.")


@st.cache_data(ttl=3600)
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


def format_number(value):
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


try:
    raw = load_data(DATA_URL)
except Exception as exc:
    st.error("영화 데이터를 불러오지 못했습니다. 데이터 주소나 인터넷 연결을 확인해 주세요.")
    st.code(str(exc))
    st.stop()

# 열 이름에 실수로 들어간 앞뒤 공백이 있어도 처리합니다.
raw.columns = [str(column).strip() for column in raw.columns]

required_columns = [
    "movieCd", "movieNm", "openDt", "genre", "nation",
    "first_scrn", "first_show", "first_date", "peak",
    "first_week_audi", "total_audi", "days_in_top10",
]
missing_columns = [column for column in required_columns if column not in raw.columns]

if missing_columns:
    st.error("CSV 파일에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.write("실제 열 이름:", ", ".join(map(str, raw.columns)))
    st.stop()

total_count = len(raw)
data = raw.copy()

for column in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# 네 가지 군집화용 열을 결측값 검사보다 먼저 모두 생성합니다.
data["롱런 지수"] = float("nan")
valid_first_week = data["first_week_audi"].notna() & (data["first_week_audi"] > 0)
data.loc[valid_first_week, "롱런 지수"] = (
    data.loc[valid_first_week, "total_audi"]
    / data.loc[valid_first_week, "first_week_audi"]
)
data.loc[data["롱런 지수"] > 20, "롱런 지수"] = 20

data["로그 스크린 수"] = data["first_scrn"].map(
    lambda value: math.log10(value)
    if pd.notna(value) and value > 0
    else float("nan")
)
data["로그 누적 관객"] = data["total_audi"].map(
    lambda value: math.log10(value)
    if pd.notna(value) and value > 0
    else float("nan")
)

# 생성 여부를 확인해 열 누락으로 인한 KeyError를 방지합니다.
missing_features = [feature for feature in FEATURES if feature not in data.columns]
if missing_features:
    st.error("분석용 속성을 만들지 못했습니다: " + ", ".join(missing_features))
    st.write("현재 열 이름:", ", ".join(map(str, data.columns)))
    st.stop()

valid_mask = (
    data["first_scrn"].notna()
    & data["total_audi"].notna()
    & data["days_in_top10"].notna()
    & data["first_week_audi"].notna()
    & (data["first_week_audi"] > 0)
    & (data["first_scrn"] > 0)
    & (data["total_audi"] > 0)
    & data[FEATURES].notna().all(axis=1)
)
data = data.loc[valid_mask].copy().reset_index(drop=True)

st.write(f"**전체 영화 {total_count:,}편 · 분석에 묶은 영화 {len(data):,}편**")

if len(data) < 3:
    st.error("분석 가능한 영화가 3편보다 적어 세 군집으로 나눌 수 없습니다.")
    st.stop()

st.subheader("1. 군집화에 사용할 속성")
selected_features = st.multiselect(
    "군집화에 사용할 속성을 두 개 이상 선택하세요. 기본값은 네 속성 전체입니다.",
    options=FEATURES,
    default=FEATURES,
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해야 합니다.")
    st.stop()

X = data[selected_features].astype(float)
X_scaled = StandardScaler().fit_transform(X)

try:
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    data["군집 번호"] = model.fit_predict(X_scaled)
except Exception as exc:
    st.error("군집화 중 오류가 발생했습니다.")
    st.code(str(exc))
    st.stop()

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰를 지정합니다.
cluster_means = data.groupby("군집 번호")["total_audi"].mean().sort_values(ascending=False)
label_map = {
    cluster_id: CLUSTER_LABELS[index]
    for index, cluster_id in enumerate(cluster_means.index.tolist())
}
data["영화 유형"] = data["군집 번호"].map(label_map)

st.subheader("2. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("가로축 속성", selected_features, index=0, key="scatter_x")
with col2:
    y_axis = st.selectbox("세로축 속성", selected_features, index=1, key="scatter_y")

fig2 = px.scatter(
    data,
    x=x_axis,
    y=y_axis,
    color="영화 유형",
    hover_name="movieNm",
    hover_data={"영화 유형": True},
    category_orders={"영화 유형": CLUSTER_LABELS},
    labels={x_axis: x_axis, y_axis: y_axis, "영화 유형": "영화 유형"},
    title="속성에 따른 영화 유형",
)
fig2.update_traces(marker={"size": 7, "opacity": 0.8})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("X축 속성", selected_features, index=0, key="scatter3_x")
    with c2:
        y3 = st.selectbox("Y축 속성", selected_features, index=1, key="scatter3_y")
    with c3:
        z3 = st.selectbox("Z축 속성", selected_features, index=2, key="scatter3_z")

    fig3 = px.scatter_3d(
        data,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        hover_name="movieNm",
        hover_data={"영화 유형": True},
        category_orders={"영화 유형": CLUSTER_LABELS},
        labels={x3: x3, y3: y3, z3: z3, "영화 유형": "영화 유형"},
        title="세 속성으로 살펴보는 영화 유형",
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 평균")
summary = (
    data.groupby("영화 유형", observed=False)
    .agg(
        편수=("movieNm", "size"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        십위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("롱런 지수", "mean"),
    )
    .reset_index()
)
summary["영화 유형"] = pd.Categorical(
    summary["영화 유형"], categories=CLUSTER_LABELS, ordered=True
)
summary = summary.sort_values("영화 유형").reset_index(drop=True)
display_summary = summary.rename(
    columns={
        "스크린_수_평균": "스크린 수 평균",
        "누적_관객_평균": "누적 관객 평균",
        "십위권_일수_평균": "10위권 일수 평균",
        "롱런_지수_평균": "롱런 지수 평균",
    }
).copy()

display_summary["편수"] = display_summary["편수"].map(lambda value: f"{value:,}")
display_summary["스크린 수 평균"] = display_summary["스크린 수 평균"].map(lambda value: f"{value:,.1f}")
display_summary["누적 관객 평균"] = display_summary["누적 관객 평균"].map(format_number)
display_summary["10위권 일수 평균"] = display_summary["10위권 일수 평균"].map(lambda value: f"{value:,.1f}")
display_summary["롱런 지수 평균"] = display_summary["롱런 지수 평균"].map(lambda value: f"{value:.2f}")
st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.caption(
    "군집화에는 선택한 속성을 표준화해 사용합니다. 표의 평균은 원래 단위이며, "
    "롱런 지수는 누적 관객을 첫 주 관객으로 나눈 뒤 20을 상한으로 적용했습니다."
)

st.subheader("5. 유형별 누적 관객 TOP 5")
for label in CLUSTER_LABELS:
    st.markdown(f"**{label} 유형**")
    top = (
        data.loc[data["영화 유형"] == label, ["movieNm", "total_audi"]]
        .sort_values("total_audi", ascending=False)
        .head(5)
        .rename(columns={"movieNm": "영화 제목", "total_audi": "누적 관객"})
        .copy()
    )
    if top.empty:
        st.write("해당 유형에 영화가 없습니다.")
    else:
        top.insert(0, "순위", range(1, len(top) + 1))
        top["누적 관객"] = top["누적 관객"].map(format_number)
        st.dataframe(top, use_container_width=True, hide_index=True)

with st.expander("분석에 사용한 영화 데이터 보기"):
    st.dataframe(
        data[
            [
                "movieNm", "영화 유형", "first_scrn", "total_audi",
                "days_in_top10", "first_week_audi", "롱런 지수",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
