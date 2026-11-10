
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="나이별 평균 공복혈당 예측기",
    page_icon="🩸",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/dyjoo2009/data/"
    "refs/heads/main/"
    "%ED%95%9C%EA%B5%AD%EC%9D%B8_%ED%98%88%EB%8B%B9_"
    "%EC%B0%B8%EC%A1%B0%ED%91%9C%EC%A4%80.csv"
)

st.title("🩸 나이별 평균 공복혈당 예측기")
st.write(
    "나이에 따른 혈당 참조값의 관계를 분석하고 "
    "선형회귀 모델로 추정합니다."
)

st.warning(
    "자료의 측정 항목은 '한국인의 혈당 농도'입니다. "
    "공복 상태에서 측정한 자료인지는 확인되지 않았으므로 "
    "결과는 공복혈당이 아닌 혈당 참조값의 추정치로 해석해야 합니다."
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    data.columns = data.columns.astype(str).str.strip()
    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# 필요한 열 확인
required_columns = [
    "지역",
    "성별",
    "나이",
    "측정값 (mg/dL)",
    "측정 수 (명)",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "CSV에서 필요한 열을 찾지 못했습니다: "
        + ", ".join(missing_columns)
    )
    st.write("실제 CSV 열 이름:", df.columns.tolist())
    st.stop()


# ==========================================
# 3. 데이터 전처리
# ==========================================

def convert_age(value):
    """
    연령대 문자열을 대표 나이로 변환합니다.

    20~24 세 -> 22
    25~26 세 -> 25.5
    75세 이상 -> 77.5 (근사값)
    """

    text = str(value).replace(" ", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return np.nan

    numbers = [float(n) for n in numbers]

    if "이상" in text:
        return numbers[0] + 2.5

    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2

    return numbers[0]


df["대표 나이"] = df["나이"].apply(convert_age)

df["측정값 (mg/dL)"] = pd.to_numeric(
    df["측정값 (mg/dL)"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce",
)

df["측정 수 (명)"] = pd.to_numeric(
    df["측정 수 (명)"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce",
)

df = df.dropna(
    subset=[
        "대표 나이",
        "측정값 (mg/dL)",
        "측정 수 (명)",
    ]
)

df = df[df["측정 수 (명)"] > 0]

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()


# ==========================================
# 4. 지역과 성별 선택
# ==========================================

st.sidebar.header("분석 조건")

regions = sorted(df["지역"].dropna().unique().tolist())

if not regions:
    st.error("지역 데이터가 없습니다.")
    st.stop()

default_region = (
    "전국" if "전국" in regions else regions[0]
)

selected_region = st.sidebar.selectbox(
    "지역 선택",
    regions,
    index=regions.index(default_region),
)

region_df = df[df["지역"] == selected_region].copy()

sexes = sorted(region_df["성별"].dropna().unique().tolist())

selected_sex = st.sidebar.selectbox(
    "성별 선택",
    ["전체(남녀 가중평균)"] + sexes,
)

if selected_sex == "전체(남녀 가중평균)":
    # 이미 별도로 계산된 전체 행이 있다면 중복 집계를 피합니다.
    sex_text = region_df["성별"].astype(str).str.strip()
    total_mask = sex_text.str.contains(
        r"전체|합계|계",
        regex=True,
    )

    if total_mask.any():
        region_df = region_df[total_mask].copy()
    else:
        region_df = region_df.copy()
else:
    region_df = region_df[
        region_df["성별"] == selected_sex
    ].copy()


if region_df.empty:
    st.error("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()


# ==========================================
# 5. 연령대별 평균 계산
# ==========================================

# 같은 연령대에 여러 행이 있으면 측정 인원수로 가중 평균
age_data = (
    region_df
    .groupby(["나이", "대표 나이"], as_index=False)
    .apply(
        lambda group: pd.Series({
            "혈당 참조값": np.average(
                group["측정값 (mg/dL)"],
                weights=group["측정 수 (명)"],
            ),
            "측정 인원": group["측정 수 (명)"].sum(),
        }),
        include_groups=False,
    )
    .reset_index(drop=True)
)

age_data = age_data.dropna(
    subset=["대표 나이", "혈당 참조값"]
)

age_data = age_data.sort_values("대표 나이")

if len(age_data) < 2:
    st.error("회귀분석을 하려면 서로 다른 연령대가 2개 이상 필요합니다.")
    st.stop()


# ==========================================
# 6. 선형회귀 모델 학습
# ==========================================

X = age_data[["대표 나이"]]
y = age_data["혈당 참조값"]
weights = age_data["측정 인원"]

model = LinearRegression()

model.fit(
    X,
    y,
    sample_weight=weights,
)

slope = float(model.coef_[0])
intercept = float(model.intercept_)

predicted = model.predict(X)

# 피어슨 상관계수
correlation = age_data["대표 나이"].corr(
    age_data["혈당 참조값"],
    method="pearson",
)

r_squared = model.score(
    X,
    y,
    sample_weight=weights,
)


# ==========================================
# 7. 주요 분석 결과
# ==========================================

st.subheader("📊 분석 결과")

col1, col2, col3 = st.columns(3)

col1.metric(
    "상관계수",
    f"{correlation:.4f}",
)

col2.metric(
    "가중 결정계수 R²",
    f"{r_squared:.4f}",
)

col3.metric(
    "분석 연령대 수",
    f"{len(age_data)}개",
)

st.markdown("### 선형회귀식")

st.latex(
    rf"\hat{{y}} = {slope:.4f}x + {intercept:.4f}"
)

st.caption(
    "x는 대표 나이, ŷ는 추정 혈당 참조값(mg/dL)입니다. "
    "회귀 모델은 측정 인원수를 가중치로 사용합니다."
)


# ==========================================
# 8. 산점도와 회귀 직선
# ==========================================

st.subheader("📈 나이와 혈당 참조값의 관계")

fig = go.Figure()

# 실제 측정값 산점도
fig.add_trace(
    go.Scatter(
        x=age_data["대표 나이"],
        y=age_data["혈당 참조값"],
        mode="markers",
        name="연령대별 측정값",
        marker=dict(
            size=11,
            opacity=0.8,
        ),
        customdata=np.column_stack([
            age_data["나이"],
            age_data["측정 인원"],
        ]),
        hovertemplate=(
            "연령대: %{customdata[0]}<br>"
            "대표 나이: %{x:.1f}세<br>"
            "혈당 참조값: %{y:.2f} mg/dL<br>"
            "측정 인원: %{customdata[1]:,.0f}명"
            "<extra></extra>"
        ),
    )
)

# 회귀 직선
line_x = np.linspace(
    age_data["대표 나이"].min(),
    age_data["대표 나이"].max(),
    200,
)

line_y = model.predict(
    pd.DataFrame({"대표 나이": line_x})
)

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형회귀 직선",
        line=dict(width=3),
    )
)

fig.update_layout(
    xaxis_title="나이 (세)",
    yaxis_title="평균 혈당 참조값 (mg/dL)",
    hovermode="closest",
    legend_title="그래프 구분",
    template="plotly_white",
)

st.plotly_chart(
    fig,
    width="stretch",
)


# ==========================================
# 9. 나이 슬라이더로 혈당값 예측
# ==========================================

st.subheader("🎚️ 나이별 혈당 참조값 추정")

min_age = float(age_data["대표 나이"].min())
max_age = float(age_data["대표 나이"].max())

if min_age == max_age:
    st.error("나이 범위가 하나뿐이어서 슬라이더를 만들 수 없습니다.")
    st.stop()

selected_age = st.slider(
    "예측할 나이를 선택하세요.",
    min_value=min_age,
    max_value=max_age,
    value=float((min_age + max_age) / 2),
    step=0.5,
    format="%.1f세",
)

estimated_glucose = float(
    model.predict(
        pd.DataFrame({
            "대표 나이": [selected_age]
        })
    )[0]
)

st.markdown("---")

st.markdown(
    f"<h3 style='text-align:center;'>"
    f"{selected_age:.1f}세의 예상 평균 혈당 참조값"
    f"</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <h1 style="
        text-align:center;
        color:#E74C3C;
        font-size:55px;
        ">
        {estimated_glucose:.2f}
        <span style="font-size:25px;">mg/dL</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ==========================================
# 10. 선택한 나이를 그래프에 표시
# ==========================================

prediction_fig = go.Figure(fig)

prediction_fig.add_trace(
    go.Scatter(
        x=[selected_age],
        y=[estimated_glucose],
        mode="markers",
        name="선택한 나이의 추정값",
        marker=dict(
            size=17,
            symbol="star",
            color="red",
        ),
    )
)

prediction_fig.update_layout(
    title=f"{selected_age:.1f}세의 추정 혈당 참조값",
)

st.plotly_chart(
    prediction_fig,
    width="stretch",
)


# ==========================================
# 11. 연령대별 데이터 표
# ==========================================

with st.expander("연령대별 분석 데이터 확인"):
    display_data = age_data.rename(
        columns={
            "나이": "연령대",
            "대표 나이": "대표 나이(세)",
            "혈당 참조값": "평균 혈당 참조값(mg/dL)",
            "측정 인원": "측정 인원(명)",
        }
    )

    st.dataframe(
        display_data,
        width="stretch",
        hide_index=True,
    )


# ==========================================
# 12. 분석 해석
# ==========================================

st.subheader("📝 분석 해석")

if correlation > 0:
    st.write(
        "나이가 증가할수록 혈당 참조값이 증가하는 "
        "양의 상관관계가 관찰됩니다."
    )
elif correlation < 0:
    st.write(
        "나이가 증가할수록 혈당 참조값이 감소하는 "
        "음의 상관관계가 관찰됩니다."
    )
else:
    st.write(
        "나이와 혈당 참조값 사이에 선형 상관관계가 거의 없습니다."
    )

st.caption(
    "주의: 상관관계는 인과관계를 의미하지 않습니다. "
    "이 모델은 연령대별 집단 자료를 이용한 추정이며, "
    "개인의 실제 혈당이나 공복혈당을 진단하는 도구가 아닙니다. "
    "75세 이상 연령대의 대표 나이는 77.5세로 근사했습니다."
)
