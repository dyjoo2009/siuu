
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="BIO-GLUCOSE LAB",
    page_icon="🧬",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/dyjoo2009/data/"
    "refs/heads/main/"
    "%ED%95%9C%EA%B5%AD%EC%9D%B8_%ED%98%88%EB%8B%B9_"
    "%EC%B0%B8%EC%A1%B0%ED%91%9C%EC%A4%80.csv"
)


# ==========================================
# 2. 디자인 CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F3F7FA;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(120deg, #102F46, #176B72);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #D8EDF0;
    font-size: 15px;
}

.section-title {
    color: #173B50;
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 13px;
    border: 1px solid #DFE9EF;
}

div[data-testid="stMetricLabel"] {
    color: #607D8B;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 8px;
    padding: 10px 18px;
}

div[data-testid="stSidebar"] {
    background-color: #EAF2F6;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. 상단 타이틀
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🧬 BIO-GLUCOSE LAB</h1>
    <p>KOREAN BLOOD GLUCOSE REFERENCE ANALYSIS SYSTEM</p>
    <p>
    연령과 성별에 따른 혈당 참조값을 분석하고
    머신러닝으로 추정하는 데이터 분석 시스템
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "본 프로그램은 한국인 집단별 혈당 참조표준을 활용한 통계적 추정 도구입니다. "
    "개인의 실제 혈당이나 당뇨병 여부를 진단하지 않습니다."
)


# ==========================================
# 4. 데이터 불러오기
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


required_columns = [
    "성별",
    "나이",
    "측정값 (mg/dL)",
    "측정 수 (명)"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()


# ==========================================
# 5. 데이터 전처리
# ==========================================

def convert_age(value):

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

for col in ["측정값 (mg/dL)", "측정 수 (명)"]:

    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "대표 나이",
        "측정값 (mg/dL)",
        "측정 수 (명)",
        "성별"
    ]
)

df = df[df["측정 수 (명)"] > 0].copy()

df["성별"] = df["성별"].astype(str).str.strip()


# ==========================================
# 6. 전국 데이터 선택
# ==========================================

if "지역" in df.columns:

    national = df[
        df["지역"].astype(str).str.strip() == "전국"
    ]

    if not national.empty:
        df = national.copy()


# ==========================================
# 7. 성별 데이터 정리
# ==========================================

# 남녀 데이터만 사용하여 전체 행의 중복 집계를 방지

sex_alias = {
    "남": "남성",
    "남자": "남성",
    "남성": "남성",
    "여": "여성",
    "여자": "여성",
    "여성": "여성"
}

df["성별"] = df["성별"].map(sex_alias)

df = df[df["성별"].isin(["남성", "여성"])].copy()

if df.empty:
    st.error(
        "남성·여성 데이터가 없습니다. "
        "CSV의 성별 컬럼 값을 확인해주세요."
    )
    st.stop()


# ==========================================
# 8. 연령대별 가중평균 계산
# ==========================================

df["가중합"] = (
    df["측정값 (mg/dL)"] * df["측정 수 (명)"]
)

age_data = (
    df.groupby(
        ["성별", "나이", "대표 나이"],
        as_index=False
    )
    .agg(
        가중합=("가중합", "sum"),
        측정인원=("측정 수 (명)", "sum")
    )
)

age_data["혈당 참조값"] = (
    age_data["가중합"] / age_data["측정인원"]
)

age_data = age_data.dropna(
    subset=["대표 나이", "혈당 참조값"]
)

age_data = age_data.sort_values(
    ["성별", "대표 나이"]
).reset_index(drop=True)


if age_data.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()


# ==========================================
# 9. 다중 선형회귀 모델
# ==========================================

# 성별: 남성 0, 여성 1
# 나이와 성별의 상호작용 항 추가
# 성별에 따라 연령과 혈당의 관계가 달라지는지도 반영

age_data["성별코드"] = (
    age_data["성별"].map({
        "남성": 0,
        "여성": 1
    })
)

age_data["나이_성별"] = (
    age_data["대표 나이"] *
    age_data["성별코드"]
)

X = age_data[
    ["대표 나이", "성별코드", "나이_성별"]
]

y = age_data["혈당 참조값"]

weights = age_data["측정인원"]

model = LinearRegression()

model.fit(
    X,
    y,
    sample_weight=weights
)

r_squared = model.score(
    X,
    y,
    sample_weight=weights
)

predictions = model.predict(X)

mae = mean_absolute_error(
    y,
    predictions,
    sample_weight=weights
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        predictions,
        sample_weight=weights
    )
)

correlations = {
    sex: group["대표 나이"].corr(
        group["혈당 참조값"]
    )
    for sex, group in age_data.groupby("성별")
}


# ==========================================
# 10. 사이드바 입력
# ==========================================

st.sidebar.title("🧬 분석 설정")

selected_sex = st.sidebar.selectbox(
    "예측할 성별",
    ["남성", "여성"]
)

min_age = float(age_data["대표 나이"].min())
max_age = float(age_data["대표 나이"].max())

if min_age >= max_age:
    st.error("분석 가능한 연령 구간이 부족합니다.")
    st.stop()

selected_age = st.sidebar.slider(
    "예측할 나이",
    min_value=min_age,
    max_value=max_age,
    value=float((min_age + max_age) / 2),
    step=0.5
)


# ==========================================
# 11. 예측 실행
# ==========================================

sex_code = 0 if selected_sex == "남성" else 1

input_data = pd.DataFrame({
    "대표 나이": [selected_age],
    "성별코드": [sex_code],
    "나이_성별": [selected_age * sex_code]
})

estimated_glucose = float(
    model.predict(input_data)[0]
)


# ==========================================
# 12. 상단 KPI
# ==========================================

st.markdown(
    '<div class="section-title">📊 모델 분석 현황</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "분석 데이터 수",
    f"{len(age_data)}개"
)

c2.metric(
    "결정계수 R²",
    f"{r_squared:.4f}"
)

c3.metric(
    "평균 절대 오차 MAE",
    f"{mae:.2f} mg/dL"
)

c4.metric(
    "RMSE",
    f"{rmse:.2f} mg/dL"
)


# ==========================================
# 13. 예측 결과
# ==========================================

st.markdown(
    '<div class="section-title">🎯 혈당 참조값 예측</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1])

with left:

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,#123B50,#197C80);
            padding:30px;
            border-radius:18px;
            color:white;
            text-align:center;
        ">
            <p style="font-size:17px;color:#D7F1F1;">
                {selected_age:.1f}세 · {selected_sex}
            </p>
            <h1 style="color:white;font-size:46px;">
                {estimated_glucose:.2f}
            </h1>
            <p style="color:#D7F1F1;">
                mg/dL
            </p>
            <p style="font-size:13px;color:#D7F1F1;">
                예상 혈당 참조값
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:

    st.markdown("#### 모델 정보")

    st.write("• 모델: 다중 선형회귀")
    st.write("• 독립변수: 나이, 성별, 나이×성별")
    st.write("• 종속변수: 혈당 참조값")
    st.write("• 학습 가중치: 측정 인원수")

    st.caption(
        "예측값은 관측된 연령 범위 내에서의 통계적 추정치입니다."
    )


# ==========================================
# 14. 탭 구성
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 연령별 분석",
    "⚖️ 남녀 비교",
    "🧠 모델 분석",
    "📋 데이터 확인"
])


# ==========================================
# TAB 1. 연령별 분석
# ==========================================

with tab1:

    st.subheader("연령에 따른 혈당 변화")

    fig = px.scatter(
        age_data,
        x="대표 나이",
        y="혈당 참조값",
        color="성별",
        size="측정인원",
        hover_data=["나이", "측정인원"],
        color_discrete_map={
            "남성": "#2387C9",
            "여성": "#E779A8"
        },
        labels={
            "대표 나이": "나이(세)",
            "혈당 참조값": "혈당 참조값(mg/dL)",
            "성별": "성별"
        },
        title="연령별 혈당 참조값 분포"
    )

    fig.update_layout(
        template="plotly_white",
        height=480
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 연령과 혈당의 상관관계")

    corr_cols = st.columns(2)

    for i, sex in enumerate(["남성", "여성"]):

        corr = correlations.get(sex, np.nan)

        with corr_cols[i]:

            st.metric(
                f"{sex} 상관계수",
                f"{corr:.4f}" if pd.notna(corr) else "계산 불가"
            )


# ==========================================
# TAB 2. 남녀 비교
# ==========================================

with tab2:

    st.subheader("남성과 여성의 혈당 참조값 비교")

    comparison = (
        age_data.groupby(
            ["나이", "대표 나이", "성별"],
            as_index=False
        )
        .apply(
            lambda g: pd.Series({
                "혈당 참조값": np.average(
                    g["혈당 참조값"],
                    weights=g["측정인원"]
                )
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    fig2 = px.line(
        comparison,
        x="대표 나이",
        y="혈당 참조값",
        color="성별",
        markers=True,
        color_discrete_map={
            "남성": "#2387C9",
            "여성": "#E779A8"
        },
        labels={
            "대표 나이": "나이(세)",
            "혈당 참조값": "혈당 참조값(mg/dL)"
        },
        title="남녀 연령별 혈당 비교"
    )

    fig2.update_layout(
        template="plotly_white",
        height=480
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.info(
        "두 성별의 혈당 차이는 관찰된 집단 참조값을 비교한 결과이며, "
        "성별이 혈당 차이의 직접적인 원인임을 의미하지 않습니다."
    )


# ==========================================
# TAB 3. 모델 분석
# ==========================================

with tab3:

    st.subheader("다중 선형회귀 모델 분석")

    st.latex(
        r"\hat{y} = b_0 + b_1A + b_2S + b_3(A \times S)"
    )

    st.write(
        "A는 나이, S는 성별 코드(남성 0, 여성 1)입니다. "
        "나이와 성별의 상호작용을 포함하여 성별에 따라 "
        "연령별 혈당 변화 기울기가 달라질 수 있도록 했습니다."
    )

    coef_df = pd.DataFrame({
        "변수": [
            "나이",
            "성별",
            "나이 × 성별"
        ],
        "회귀계수": model.coef_
    })

    st.dataframe(
        coef_df,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "회귀모델 절편",
        f"{model.intercept_:.4f}"
    )

    st.metric(
        "가중 결정계수",
        f"{r_squared:.4f}"
    )

    st.metric(
        "가중 MAE",
        f"{mae:.3f} mg/dL"
    )

    st.caption(
        "현재 성능 지표는 학습 데이터에 대한 평가 결과입니다. "
        "새로운 데이터에 대한 예측 성능을 보장하지 않으므로 "
        "별도의 테스트 데이터 평가가 필요합니다."
    )


# ==========================================
# TAB 4. 데이터 확인
# ==========================================

with tab4:

    st.subheader("분석에 사용한 데이터")

    st.dataframe(
        age_data[
            [
                "성별",
                "나이",
                "대표 나이",
                "혈당 참조값",
                "측정인원"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    csv = age_data.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 분석 데이터 다운로드",
        data=csv,
        file_name="blood_glucose_analysis.csv",
        mime="text/csv"
    )


# ==========================================
# 15. 생화학 탐구 연결
# ==========================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🧪 생화학적 탐구와의 연결</div>',
    unsafe_allow_html=True
)

bio1, bio2, bio3 = st.columns(3)

with bio1:

    st.markdown("### 🧬 포도당 대사")

    st.write(
        "포도당은 세포에서 ATP를 생성하는 주요 에너지원으로 "
        "사용되며, 혈당은 에너지 대사와 밀접한 관계가 있습니다."
    )

with bio2:

    st.markdown("### 🔬 인슐린 작용")

    st.write(
        "인슐린은 세포의 포도당 흡수를 촉진하여 "
        "혈당 조절에 관여하는 호르몬입니다."
    )

with bio3:

    st.markdown("### 🧫 대사 변화")

    st.write(
        "연령과 성별에 따른 혈당 참조값의 차이를 분석하여 "
        "인체의 대사 특성에 관한 탐구로 연결할 수 있습니다."
    )


# ==========================================
# 16. 하단 안내
# ==========================================

st.markdown("---")

st.caption(
    "BIO-GLUCOSE LAB | Korean Blood Glucose Reference Analysis"
)

st.caption(
    "본 프로그램은 집단별 혈당 참조표준을 이용한 교육·탐구용 모델입니다. "
    "개인의 혈당 진단이나 의학적 판단에 사용하지 마세요."
)
