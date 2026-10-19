import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 규모와 롱런 특성을 바탕으로 비슷한 영화끼리 묶어 봅니다.")

FEATURES = [
    "로그 스크린 수",
    "로그 누적 관객",
    "10위권 일수",
    "롱런 지수"
]
ORIGINAL_FEATURES = ["first_scrn", "total_audi", "days_in_top10", "롱런 지수"]
FEATURE_LABELS = {
    "로그 스크린 수": "로그 스크린 수",
    "로그 누적 관객": "로그 누적 관객",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수"
}
AVERAGE_LABELS = {
    "first_scrn": "스크린 수",
    "total_audi": "누적 관객",
    "days_in_top10": "10위권 일수",
    "롱런 지수": "롱런 지수"
}

@st.cache_data(ttl=3600)
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    return df

try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 데이터 주소나 인터넷 연결을 확인해 주세요.")
    st.exception(e)
    st.stop()

required_columns = [
    "movieCd", "movieNm", "openDt", "genre", "nation",
    "first_scrn", "first_show", "first_date", "peak",
    "first_week_audi", "total_audi", "days_in_top10"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

total_movies = len(df)
work = df.copy()

numeric_columns = ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]
for col in numeric_columns:
    work[col] = pd.to_numeric(work[col], errors="coerce")

# 로그를 취하는 두 값은 양수여야 하며, 첫 주 관객은 0이면 제외합니다.
work["롱런 지수"] = work["total_audi"] / work["first_week_audi"]
work.loc[work["롱런 지수"] > 20, "롱런 지수"] = 20
work["로그 스크린 수"] = work["first_scrn"].where(work["first_scrn"] > 0).map(
    lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan")
)
work["로그 누적 관객"] = work["total_audi"].where(work["total_audi"] > 0).map(
    lambda x: __import__("math").log10(x) if pd.notna(x) else float("nan")
)

# 군집화에 쓸 네 속성 중 결측이 있거나 첫 주 관객이 0인 영화는 제외합니다.
valid_mask = (
    work["first_scrn"].notna()
    & work["total_audi"].notna()
    & work["days_in_top10"].notna()
    & work["first_week_audi"].notna()
    & (work["first_week_audi"] > 0)
    & (work["first_scrn"] > 0)
    & (work["total_audi"] > 0)
    & work[FEATURES].notna().all(axis=1)
)
data = work.loc[valid_mask].copy().reset_index(drop=True)
st.write(f"**전체 영화 {total_movies:,}편 · 분석에 묶은 영화 {len(data):,}편**")

if len(data) < 3:
    st.error("군집화를 하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.subheader("1. 군집화에 사용할 속성")
selected_features = st.multiselect(
    "두 개 이상 선택하세요. 기본값은 네 속성 전체입니다.",
    options=FEATURES,
    default=FEATURES
)

if len(selected_features) < 2:
    st.warning("k-평균 군집화를 하려면 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# 선택 속성 표준화 후 고정된 난수 시드로 k-평균 군집화
X = data[selected_features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
data["군집 번호"] = model.fit_predict(X_scaled)

# 군집 번호는 임의적이므로, 원래 단위의 누적 관객 평균 순서로 이름을 붙입니다.
cluster_order = (
    data.groupby("군집 번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_names = {cluster_id: ["㉮", "㉯", "㉰"][rank] for rank, cluster_id in enumerate(cluster_order)}
data["영화 유형"] = data["군집 번호"].map(cluster_names)

st.subheader("2. 2차원 산점도")
axis_options = selected_features
axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_axis = st.selectbox("가로축 속성", axis_options, index=0, key="x_axis")
with axis_col2:
    y_default = 1 if len(axis_options) > 1 else 0
    y_axis = st.selectbox("세로축 속성", axis_options, index=y_default, key="y_axis")

fig2 = px.scatter(
    data,
    x=x_axis,
    y=y_axis,
    color="영화 유형",
    hover_name="movieNm",
    hover_data={"영화 유형": True, x_axis: ":.3f", y_axis: ":.3f"},
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    labels={x_axis: x_axis, y_axis: y_axis, "영화 유형": "영화 유형"},
    title=f"{x_axis}과(와) {y_axis}으로 본 영화 유형"
)
fig2.update_traces(marker={"size": 8, "opacity": 0.8})
fig2.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("X축 속성", selected_features, index=0, key="x3")
    with c2:
        y3 = st.selectbox("Y축 속성", selected_features, index=1, key="y3")
    with c3:
        z3 = st.selectbox("Z축 속성", selected_features, index=2, key="z3")

    fig3 = px.scatter_3d(
        data,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        hover_name="movieNm",
        hover_data={"영화 유형": True, x3: ":.3f", y3: ":.3f", z3: ":.3f"},
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        labels={x3: x3, y3: y3, z3: z3, "영화 유형": "영화 유형"},
        title="세 속성으로 살펴보는 영화 유형"
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형")
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 요약")
summary = data.groupby("영화 유형").agg(
    편수=("movieNm", "size"),
    스크린_수_평균=("first_scrn", "mean"),
    누적_관객_평균=("total_audi", "mean"),
    십위권_일수_평균=("days_in_top10", "mean"),
    롱런_지수_평균=("롱런 지수", "mean")
).reset_index()

summary = summary.rename(columns={
    "스크린_수_평균": "스크린 수 평균",
    "누적_관객_평균": "누적 관객 평균",
    "십위권_일수_평균": "10위권 일수 평균",
    "롱런_지수_평균": "롱런 지수 평균"
})
summary["영화 유형"] = pd.Categorical(summary["영화 유형"], categories=["㉮", "㉯", "㉰"], ordered=True)
summary = summary.sort_values("영화 유형").reset_index(drop=True)

display_summary = summary.copy()
display_summary["스크린 수 평균"] = display_summary["스크린 수 평균"].round(1)
display_summary["누적 관객 평균"] = display_summary["누적 관객 평균"].round(0).map(lambda x: f"{x:,.0f}")
display_summary["10위권 일수 평균"] = display_summary["10위권 일수 평균"].round(1)
display_summary["롱런 지수 평균"] = display_summary["롱런 지수 평균"].round(2)
display_summary["편수"] = display_summary["편수"].map(lambda x: f"{x:,}")
st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.caption("로그 변환은 군집화에만 사용하며, 위 표의 평균은 원래 단위 기준입니다. 롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며 20을 초과하면 20으로 제한했습니다.")

st.subheader("5. 유형별 누적 관객 TOP 5")
for label in ["㉮", "㉯", "㉰"]:
    group = data[data["영화 유형"] == label].sort_values("total_audi", ascending=False).head(5)
    st.markdown(f"**{label} 유형**")
    if group.empty:
        st.write("해당 유형에 영화가 없습니다.")
    else:
        top = group[["movieNm", "total_audi"]].rename(
            columns={"movieNm": "영화 제목", "total_audi": "누적 관객"}
        ).copy()
        top["누적 관객"] = top["누적 관객"].map(lambda x: f"{x:,.0f}")
        top.insert(0, "순위", range(1, len(top) + 1))
        st.dataframe(top, use_container_width=True, hide_index=True)

with st.expander("분석에 사용한 데이터 확인"):
    st.write("군집화에 사용한 속성:", ", ".join(selected_features))
    st.dataframe(
        data[["movieNm", "영화 유형"] + selected_features + ["first_scrn", "total_audi", "first_week_audi", "days_in_top10"]],
        use_container_width=True,
        hide_index=True
    )
