
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="BIO-GLUCOSE LAB",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
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
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
    }

    .stApp {
        background-color: #F3F7FA;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(120deg, #102D46, #176B72);
        padding: 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        color: #D5E9EC;
        font-size: 15px;
        margin: 0;
    }

    .section-title {
        color: #153D52;
        font-size: 21px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #E2EBF0;
        box-shadow: 0 3px 12px rgba(20,50,70,0.04);
    }

    .small-label {
        color: #718596;
        font-size: 13px;
        font-weight: 600;
    }

    .big-number {
        color: #176B72;
        font-size: 29px;
        font-weight: 800;
    }

    .notice {
        background: #E8F4F5;
        color: #205A61;
        padding: 15px;
        border-radius: 12px;
        font-size: 13px;
        line-height: 1.7;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF1F5;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 17px;
        border-radius: 14px;
        border: 1px solid #E2EBF0;
    }

    h2, h3 {
        color: #153D52;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. 제목
# ==========================================

st.markdown("""
<div class="hero">
    <div style="font-size:14px; letter-spacing:3px;">
        BIOCHEMISTRY · DATA SCIENCE · HEALTH
    </div>
    <h1>🧬 BIO-GLUCOSE LAB</h1>
    <p>한국인 연령·성별별 혈당 참조값 분석 및 예측 시스템</p>
    <p style="margin-top:14px;">
        AGE & GENDER BASED GLUCOSE REFERENCE PREDICTION
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
    <b>연구 안내</b><br>
    본 프로그램은 한국인 혈당 참조표준의 집단별 자료를 이용하여
    연령과 성별에 따른 혈당 참조값을 분석하고 추정합니다.
    개인의 실제 혈당이나 당뇨병 여부를 진단하는 의료기기가 아닙니다.
    또한 원자료의 공복 여부는 확인되지 않았습니다.
</div>
""", unsafe_allow_html=True)


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
    "지역",
    "성별",
    "나이",
    "측정값 (mg/dL)",
    "측정 수 (명)",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error("필요한 컬럼이 없습니다: " + ", ".join(missing))
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

df["측정값 (mg/dL)"] = pd.to_numeric(
    df["측정값 (mg/dL)"].astype(str).str.replace(",", ""),
    errors="coerce",
)

df["측정 수 (명)"] = pd.to_numeric(
    df["측정 수 (명)"].astype(str).str.replace(",", ""),
    errors="coerce",
)

df["성별"] = df["성별"].astype(str).str.strip()
df["지역"] = df["지역"].astype(str).str.strip()


df = df.dropna(
    subset=[
        "대표 나이",
        "측정값 (mg/dL)",
        "측정 수 (명)",
        "성별",
    ]
)

df = df[df["측정 수 (명)"] > 0]


# 전국 데이터 선택
national_mask = df["지역"] == "전국"

if national_mask.any():

    df = df[national_mask].copy()

else:

    st.error(
        "CSV에서 전국 자료를 찾지 못했습니다. "
        "지역별 자료를 임의로 합치지 않도록 분석을 중단합니다."
    )
    st.stop()


# 전체 또는 합계 행 제거
df = df[
    ~df["성별"].str.contains(
        r"전체|합계|계",
        regex=True,
        na=False,
    )
].copy()


# 성별 데이터 확인
gender_values = sorted(df["성별"].dropna().unique().tolist())

if len(gender_values) != 2:

    st.error(
        "성별 데이터가 정확히 2개인지 확인해주세요. "
        f"현재 확인된 값: {gender_values}"
    )
    st.stop()


if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()


# ==========================================
# 6. 연령대별 가중평균 계산
# ==========================================

df["가중합"] = (
    df["측정값 (mg/dL)"] * df["측정 수 (명)"]
)

age_gender_data = (
    df.groupby(
        ["나이", "대표 나이", "성별"],
        as_index=False
    )
    .agg(
        가중합=("가중합", "sum"),
        측정인원=("측정 수 (명)", "sum"),
    )
)

age_gender_data["혈당
