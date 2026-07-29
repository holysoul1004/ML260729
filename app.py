import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==========================================
# 0. 페이지 기본 설정 및 세션 초기화
# ==========================================
st.set_page_config(
    page_title="CSV 데이터로 배우는 선형회귀 실험실",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 상태 초기화
if "df" not in st.session_state:
    st.session_state["df"] = None
if "simple_model_res" not in st.session_state:
    st.session_state["simple_model_res"] = None
if "multi_model_res" not in st.session_state:
    st.session_state["multi_model_res"] = None


# ==========================================
# 1. 헬퍼 함수 정의
# ==========================================
def generate_sample_data(n_samples=120):
    """미세먼지 예제 데이터 생성 함수 (최소 100행 이상)"""
    np.random.seed(42)
    temp = np.random.uniform(5, 35, n_samples)
    humidity = np.random.uniform(30, 90, n_samples)
    wind_speed = np.random.uniform(0.5, 8.0, n_samples)
    rainfall = np.random.choice(
        [0, 0, 0, 0, 2, 5, 12, 25], size=n_samples, p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.04, 0.04, 0.02]
    )

    # 물리적 직관을 반영한 물리 방정식 + 노이즈
    # Wind speed -> PM2.5 감소, Temp -> 화학반응 증가, Humidity -> 미세먼지 축적
    pm25 = (
        45.0
        + (0.8 * temp)
        + (0.3 * humidity)
        - (4.5 * wind_speed)
        - (1.2 * rainfall)
        + np.random.normal(0, 8, n_samples)
    )
    pm25 = np.clip(pm25, 5, 150)  # 음수 방지

    df = pd.DataFrame(
        {
            "temperature": np.round(temp, 1),
            "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 1),
            "rainfall": np.round(rainfall, 1),
            "pm25": np.round(pm25, 1),
        }
    )
    return df


def load_csv(uploaded_file):
    """UTF-8 및 CP949 인코딩 처리하여 CSV 읽기"""
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="cp949")
    except Exception as e:
        raise Exception(f"CSV 파일 읽기 실패: {str(e)}")


def validate_data(df):
    """데이터 유효성 검사"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    warnings = []
    if len(df) < 10:
        warnings.append("데이터 수가 10개 미만입니다. 모델 학습이 불가능합니다.")
    elif len(df) < 30:
        warnings.append(
            "데이터 수가 30개 미만으로 적습니다. 회귀 분석 결과 해석 시 주의하세요."
        )

    if len(numeric_cols) < 2:
        warnings.append("선형회귀를 수행하기 위해서는 최소 2개 이상의 숫자형 열이 필요합니다.")

    return numeric_cols, categorical_cols, warnings


def calculate_metrics(y_true, y_pred, n_features):
    """회귀 평가 지표 계산"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    if n - n_features - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = r2

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2,
    }


def train_simple_regression(df, x_col, y_col, test_size):
    """단순선형회귀 모델 학습 및 결과 반환"""
    data = df[[x_col, y_col]].dropna()
    X = data[[x_col]]
    y = data[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, n_features=1)
    corr = data[x_col].corr(data[y_col])

    return {
        "model": model,
        "x_col": x_col,
        "y_col": y_col,
        "coef": model.coef_[0],
        "intercept": model.intercept_,
        "metrics": metrics,
        "corr": corr,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def train_multiple_regression(df, x_cols, y_col, test_size, use_scaler):
    """다중선형회귀 모델 학습 및 결과 반환"""
    data = df[x_cols + [y_col]].dropna()
    X = data[x_cols]
    y = data[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if use_scaler:
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        pipeline.fit(X_train, y_train)
        model = pipeline.named_steps["model"]
        scaler = pipeline.named_steps["scaler"]
        y_pred = pipeline.predict(X_test)
        coefs = model.coef_
        intercept = model.intercept_
    else:
        pipeline = None
        scaler = None
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        coefs = model.coef_
        intercept = model.intercept_

    metrics = calculate_metrics(y_test, y_pred, n_features=len(x_cols))

    return {
        "model": model,
        "pipeline": pipeline,
        "scaler": scaler,
        "use_scaler": use_scaler,
        "x_cols": x_cols,
        "y_col": y_col,
        "coefs": coefs,
        "intercept": intercept,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ==========================================
# 2. 사이드바 구성
# ==========================================
with st.sidebar:
    st.title("🔬 AI 기초 실험실")
    st.write("**주제:** 선형회귀 모델 탐구")
    st.markdown("---")
    st.subheader("💡 핵심 개념 요약")
    st.markdown(
        """
    - **독립변수(X):** 원인이 되는 변수
    - **종속변수(y):** 결과가 되는 변수
    - **회귀계수:** X가 1 변화할 때 y의 변화량
    - **잔차:** 실제값과 예측값의 차이
    - **$R^2$:** 모델의 설명력 (1에 가까울수록 좋음)
    """
    )
    st.markdown("---")
    st.caption("고등학교 '인공지능 기초' 수업용")


# ==========================================
# 3. 메인 인터페이스 및 탭 설정
# ==========================================
st.title("📊 CSV 데이터로 배우는 선형회귀 실험실")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1️⃣ 학습 안내",
        "2️⃣ CSV 데이터 업로드",
        "3️⃣ 데이터 탐색",
        "4️⃣ 단순선형회귀",
        "5️⃣ 다중선형회귀",
        "6️⃣ 모델 평가 및 비교",
    ]
)


# ------------------------------------------
# TAB 1: 학습 안내
# ------------------------------------------
with tab1:
    st.header("📘 선형회귀 개념 다지기")

    col1, col2 = st.columns(2)

    with col1:
        st.info("💡 **회귀(Regression)란?**")
        st.write(
            "여러 변수 사이의 관계를 파악하여, 한 변수의 값으로 다른 변수의 값을 **예측**하는 통계적 및 인공지능 기법입니다."
        )

        st.info("🎯 **독립변수(X) vs 종속변수(y)**")
        st.write("- **독립변수(X):** 영향을 주는 변수 (예: 기온, 풍속, 학습 시간)")
        st.write("- **종속변수(y):** 영향을 받는 변수 (예: 미세먼지 농도, 시험 점수)")

    with col2:
        st.info("📈 **선형회귀(Linear Regression)란?**")
        st.write(
            "변수들 사이의 관계를 **직선 형태(선형 관계)**로 모델링하는 가장 기초적이고 강력한 인공지능 알고리즘입니다."
        )

        st.warning("⚠️ **주의! 상관관계 $\neq$ 인과관계**")
        st.write(
            "두 변수가 함께 증가하거나 감소하더라도(상관관계), 하나가 다른 하나의 직접적인 원인(인과관계)이 아닐 수 있습니다."
        )

    st.markdown("---")
    st.subheader("📐 수학적 표현식")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 1. 단순선형회귀 (Simple Linear Regression)")
        st.write("독립변수(X)가 **1개**일 때의 모델입니다.")
        st.latex(r"\hat{y} = b_0 + b_1 x")
        st.caption("($b_0$: Y절편, $b_1$: 기울기/회귀계수, $\hat{y}$: 예측값)")

    with c2:
        st.markdown("##### 2. 다중선형회귀 (Multiple Linear Regression)")
        st.write("독립변수(X)가 **2개 이상**일 때의 모델입니다.")
        st.latex(r"\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n")
        st.caption("($b_0$: Y절편, $b_1, b_2, \dots$: 각 변수의 회귀계수)")

    st.markdown("---")
    st.subheader("🔍 실제값, 예측값, 잔차란?")
    st.write(
        "- **실제값 ($y$):** 데이터셋에 실제로 기록된 관측치입니다.\n"
        "- **예측값 ($\hat{y}$):** 회귀선(모델)이 X값을 바탕으로 계산해낸 예측치입니다.\n"
        "- **잔차 (Residual, $e = y - \hat{y}$):** 실제값과 예측값의 오차입니다. 인공지능은 이 잔차의 제곱합을 최소화하도록 학습합니다."
    )


# ------------------------------------------
# TAB 2: CSV 데이터 업로드
# ------------------------------------------
with tab2:
    st.header("📂 CSV 데이터 준비하기")

    col_up, col_dn = st.columns([2, 1])

    with col_dn:
        st.subheader("📥 예제 데이터 내려받기")
        sample_df = generate_sample_data(120)
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="🌡️ 미세먼지 예제 CSV 다운로드",
            data=csv_bytes,
            file_name="seoul_weather_pm25.csv",
            mime="text/csv",
            help="기온, 습도, 풍속, 강수량, PM2.5 미세먼지 데이터가 포함된 예제 파일입니다.",
        )
        st.caption("예제 데이터: 기상 요인에 따른 PM2.5 미세먼지 농도 (120행)")

    with col_up:
        st.subheader("📤 나의 CSV 파일 업로드")
        file = st.file_uploader("CSV 파일을 드래그하여 놓거나 선택하세요", type=["csv"])

        if file is not None:
            try:
                df = load_csv(file)
                st.session_state["df"] = df
                st.success("🎉 성공적으로 파일이 업로드되었습니다!")
            except Exception as e:
                st.error(f"❌ 파일을 읽는 도중 오류가 발생했습니다: {e}")
        elif st.session_state["df"] is None:
            st.session_state["df"] = sample_df
            st.info("💡 업로드된 파일이 없어 기본 예제 데이터(미세먼지 예제)가 설정되었습니다.")

    st.markdown("---")

    df = st.session_state["df"]
    if df is not None:
        st.subheader("👀 데이터 미리보기 및 요약 정보")

        st.dataframe(df.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 행(Row) 수", f"{df.shape[0]} 개")
        c2.metric("전체 열(Column) 수", f"{df.shape[1]} 개")

        numeric_cols, categorical_cols, warnings = validate_data(df)
        c3.metric("숫자형 변수 수", f"{len(numeric_cols)} 개")
        c4.metric("문자형 변수 수", f"{len(categorical_cols)} 개")

        for w in warnings:
            st.warning(f"⚠️ {w}")

        st.markdown("##### 📌 열별 데이터 타입 및 결측값 현황")
        info_df = pd.DataFrame(
            {
                "데이터 타입": df.dtypes.astype(str),
                "결측값(Null) 개수": df.isnull().sum(),
                "결측 비율(%)": np.round(df.isnull().sum() / len(df) * 100, 1),
            }
        )
        st.dataframe(info_df.T, use_container_width=True)


# ------------------------------------------
# TAB 3: 데이터 탐색
# ------------------------------------------
with tab3:
    st.header("🔍 데이터 탐색 (EDA)")
    df = st.session_state["df"]

    if df is None:
        st.warning("데이터를 먼저 업로드하거나 예제 데이터를 확인해주세요.")
    else:
        numeric_cols, _, _ = validate_data(df)

        if len(numeric_cols) < 2:
            st.error("데이터 탐색 및 회귀 분석을 진행하려면 숫자형 열이 최소 2개 필요합니다.")
        else:
            st.subheader("📊 1. 숫자형 변수 기술통계량")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

            st.markdown("---")
            st.subheader("📈 2. 변수별 분포 및 산점도 확인")

            c_x, c_y = st.columns(2)
            with c_x:
                x_var = st.selectbox(
                    "X축 변수 선택", numeric_cols, index=2 if "wind_speed" in numeric_cols else 0
                )
            with c_y:
                default_y_idx = (
                    numeric_cols.index("pm25")
                    if "pm25" in numeric_cols
                    else min(1, len(numeric_cols) - 1)
                )
                y_var = st.selectbox("Y축 변수 선택", numeric_cols, index=default_y_idx)

            col_hist, col_scatter = st.columns(2)

            with col_hist:
                fig_hist = px.histogram(
                    df,
                    x=x_var,
                    title=f"[{x_var}] 히스토그램 (분포)",
                    marginal="box",
                    color_discrete_sequence=["#4C78A8"],
                )
                fig_hist.update_layout(xaxis_title=x_var, yaxis_title="빈도수")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_scatter:
                fig_scatter = px.scatter(
                    df,
                    x=x_var,
                    y=y_var,
                    title=f"[{x_var}] vs [{y_var}] 산점도",
                    hover_data=df.columns,
                    color_discrete_sequence=["#F28E2B"],
                )
                fig_scatter.update_layout(
                    xaxis_title=f"{x_var} (독립변수 후보)",
                    yaxis_title=f"{y_var} (종속변수 후보)",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")
            st.subheader("🔥 3. 상관계수 분석")

            corr_matrix = df[numeric_cols].corr()

            c_corr_table, c_corr_map = st.columns([1, 1.2])

            with c_corr_table:
                st.write("**상관계수 표 (Pearson Correlation)**")
                st.dataframe(
                    corr_matrix.style.background_gradient(cmap="coolwarm").format("{:.2f}"),
                    use_container_width=True,
                )

            with c_corr_map:
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="상관계수 히트맵",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            st.info(
                """
            💡 **상관계수(r) 해석 가이드**
            - **+1.0에 가까움:** 강한 양의 상관관계 (X가 늘면 Y도 증가)
            - **-1.0에 가까움:** 강한 음의 상관관계 (X가 늘면 Y는 감소)
            - **0.0 근처:** 선형적 상관관계가 거의 없음
            """
            )

            with st.expander("❓ [탐구 질문 1] 산점도와 상관계수 관찰하기"):
                st.markdown(
                    """
                1. 선택한 두 변수는 **양의 관계**인가요, **음의 관계**인가요?
                2. 데이터 점들이 하나 직선 주변에 촘촘하게 모여 있나요, 아니면 넓게 흩어져 있나요?
                3. 다른 점들과 멀리 떨어져 있는 **이상치(Outlier)**로 보이는 데이터가 있나요?
                4. 두 변수의 상관계수가 높다면, 한 변수가 다른 변수의 **직접적 원인**이라고 단정할 수 있을까요?
                """
                )


# ------------------------------------------
# TAB 4: 단순선형회귀
# ------------------------------------------
with tab4:
    st.header("1️⃣ 단순선형회귀 (Simple Linear Regression)")
    df = st.session_state["df"]

    if df is None:
        st.warning("데이터를 먼저 업로드해 주세요.")
    else:
        numeric_cols, _, warnings = validate_data(df)
        if len(numeric_cols) < 2:
            st.error("단순선형회귀 분석을 위해 최소 2개 이상의 숫자형 열이 필요합니다.")
        else:
            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                x_col = st.selectbox(
                    "독립변수 (X) 선택",
                    numeric_cols,
                    index=2 if "wind_speed" in numeric_cols else 0,
                    key="simple_x",
                )
            with c2:
                y_options = [c for c in numeric_cols if c != x_col]
                default_y_idx = (
                    y_options.index("pm25")
                    if "pm25" in y_options
                    else min(0, len(y_options) - 1)
                )
                y_col = st.selectbox(
                    "종속변수 (y) 선택", y_options, index=default_y_idx, key="simple_y"
                )
            with c3:
                test_size = st.slider(
                    "테스트 데이터 비율 (Test Size)",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05,
                    key="simple_test_size",
                )

            res = train_simple_regression(df, x_col, y_col, test_size)
            st.session_state["simple_model_res"] = res

            st.markdown("---")
            st.subheader("📝 1. 모델 학습 결과 및 수식")

            eq_str = f"예측 {y_col} = {res['coef']:.4f} × {x_col} + ({res['intercept']:.4f})"
            st.success(f"**학습된 단순회귀식:**  \n### `{eq_str}`")

            m = res["metrics"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("학습 데이터 수", f"{len(res['X_train'])}개")
            mc2.metric("테스트 데이터 수", f"{len(res['X_test'])}개")
            mc3.metric("상관계수 (r)", f"{res['corr']:.3f}")
            mc4.metric("결정계수 (R²)", f"{m['R2']:.3f}")
            mc5.metric("RMSE (평균오차)", f"{m['RMSE']:.3f}")

            # 자동 해석 메시지
            direction = "증가" if res["coef"] > 0 else "감소"
            st.info(
                f"💡 **기울기 해석:** `{x_col}` 변수가 1 단위 증가할 때, `{y_col}` 예측값은 평균적으로 약 **{abs(res['coef']):.3f}** 만큼 **{direction}**하는 경향을 보입니다."
            )

            st.markdown("---")
            st.subheader("📈 2. 회귀선 및 잔차 시각화")

            # 시각화 데이터 생성
            X_all = df[[x_col]].dropna()
            x_range = np.linspace(X_all.min(), X_all.max(), 100)
            y_range = res["model"].predict(x_range)

            fig_reg = go.Figure()
            # Train Data (opacity=0.7로 수정)
            fig_reg.add_trace(
                go.Scatter(
                    x=res["X_train"][x_col],
                    y=res["y_train"],
                    mode="markers",
                    name="학습 데이터 (Train)",
                    marker=dict(color="#1f77b4", opacity=0.7),
                )
            )
            # Test Data
            fig_reg.add_trace(
                go.Scatter(
                    x=res["X_test"][x_col],
                    y=res["y_test"],
                    mode="markers",
                    name="테스트 데이터 (Test)",
                    marker=dict(color="#ff7f0e", symbol="diamond", size=8),
                )
            )
            # Regression Line
            fig_reg.add_trace(
                go.Scatter(
                    x=x_range.flatten(),
                    y=y_range,
                    mode="lines",
                    name="선형 회귀선",
                    line=dict(color="red", width=2),
                )
            )

            fig_reg.update_layout(
                title=f"[{x_col}] 과 [{y_col}] 의 회귀 분석",
                xaxis_title=x_col,
                yaxis_title=y_col,
            )

            st.plotly_chart(fig_reg, use_container_width=True)

            # 잔차 선 표시 시각화
            st.markdown("##### 📏 테스트 데이터의 잔차(Residual) 시각화")
            fig_res = go.Figure()
            fig_res.add_trace(
                go.Scatter(
                    x=res["X_test"][x_col],
                    y=res["y_test"],
                    mode="markers",
                    name="실제값 (Test)",
                    marker=dict(color="#ff7f0e"),
                )
            )
            fig_res.add_trace(
                go.Scatter(
                    x=res["X_test"][x_col],
                    y=res["y_pred"],
                    mode="markers",
                    name="예측값",
                    marker=dict(color="black", symbol="x"),
                )
            )

            # 잔차 실선 추가
            for idx in range(len(res["X_test"])):
                x_val = res["X_test"][x_col].iloc[idx]
                y_true_val = res["y_test"].iloc[idx]
                y_pred_val = res["y_pred"][idx]
                fig_res.add_shape(
                    type="line",
                    x0=x_val,
                    y0=y_true_val,
                    x1=x_val,
                    y1=y_pred_val,
                    line=dict(color="gray", width=1, dash="dot"),
                )

            fig_res.update_layout(
                title="테스트 데이터 점과 회귀선 사이의 수직 오차(잔차)",
                xaxis_title=x_col,
                yaxis_title=y_col,
            )
            st.plotly_chart(fig_res, use_container_width=True)

            st.markdown("---")
            st.subheader("🔮 3. 새로운 데이터 예측해보기")

            x_min_val = float(df[x_col].min())
            x_max_val = float(df[x_col].max())
            x_mean_val = float(df[x_col].mean())

            new_x = st.number_input(
                f"새로운 `{x_col}` 값 입력:",
                min_value=x_min_val - (x_max_val - x_min_val),
                max_value=x_max_val + (x_max_val - x_min_val),
                value=x_mean_val,
            )

            pred_y = res["model"].predict([[new_x]])[0]
            st.metric(
                label=f"🎯 예측된 `{y_col}` 값", value=f"{pred_y:.2f}"
            )

            if pred_y < 0:
                st.warning(
                    f"⚠️ **참고:** 선형회귀 모델이 음수값({pred_y:.2f})을 예측했습니다. 물리적으로 `{y_col}`은(는) 음수가 될 수 없지만, 이는 선형 모델의 한계점 중 하나입니다."
                )

            st.caption(
                "“이 값은 데이터에서 학습한 선형적인 경향을 이용한 예측값이며 실제값과 다를 수 있습니다.”"
            )

            with st.expander("❓ [탐구 질문 2 & 3 & 4] 회귀선과 잔차 이해하기"):
                st.markdown(
                    """
                1. **회귀선은 모든 데이터 점을 통과하나요?** 그렇지 않다면 그 이유는 무엇일까요?
                2. **기울기의 부호(양수/음수)**는 두 변수 사이의 어떤 관점에서의 방향을 의미할까요?
                3. **잔차가 양수(+)인 경우**는 실제값과 예측값 중 어느 것이 더 큰 상태인가요?
                """
                )


# ------------------------------------------
# TAB 5: 다중선형회귀
# ------------------------------------------
with tab5:
    st.header("2️⃣ 다중선형회귀 (Multiple Linear Regression)")
    df = st.session_state["df"]

    if df is None:
        st.warning("데이터를 먼저 업로드해 주세요.")
    else:
        numeric_cols, _, _ = validate_data(df)

        if len(numeric_cols) < 3:
            st.error(
                "다중선형회귀 분석을 수행하려면 최소 3개 이상의 숫자형 변수가 필요합니다."
            )
        else:
            col_sel1, col_sel2 = st.columns([2, 1])

            with col_sel2:
                default_y_idx = (
                    numeric_cols.index("pm25")
                    if "pm25" in numeric_cols
                    else len(numeric_cols) - 1
                )
                y_col_multi = st.selectbox(
                    "종속변수 (y) 선택",
                    numeric_cols,
                    index=default_y_idx,
                    key="multi_y",
                )

            x_options = [c for c in numeric_cols if c != y_col_multi]

            with col_sel1:
                default_x = (
                    x_options
                    if len(x_options) >= 2
                    else x_options
                )
                x_cols_multi = st.multiselect(
                    "독립변수들 (X) 선택 (최소 2개 이상 선택):",
                    options=x_options,
                    default=default_x,
                    key="multi_x",
                )

            c_scale, c_test = st.columns(2)
            with c_scale:
                use_scaler = st.checkbox(
                    "변수 표준화(StandardScaler) 적용",
                    value=False,
                    help="변수별 단위가 다를 때 표준화(평균 0, 표준편차 1)를 수행하여 계수 크기를 단순 비교할 수 있게 합니다.",
                )
            with c_test:
                test_size_multi = st.slider(
                    "테스트 데이터 비율",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05,
                    key="multi_test_size",
                )

            if len(x_cols_multi) < 2:
                st.warning("⚠️ 다중선형회귀를 실행하려면 독립변수(X)를 최소 2개 이상 선택해주세요.")
            else:
                res_multi = train_multiple_regression(
                    df, x_cols_multi, y_col_multi, test_size_multi, use_scaler
                )
                st.session_state["multi_model_res"] = res_multi

                st.markdown("---")
                st.subheader("📝 1. 모델 학습 결과 및 다중회귀식")

                # 회귀식 문자열 구성
                terms = [f"({b:.4f} × {col})" for b, col in zip(res_multi["coefs"], x_cols_multi)]
                eq_multi = f"예측 {y_col_multi} = {' + '.join(terms)} + ({res_multi['intercept']:.4f})"
                st.success(f"**학습된 다중회귀식:**  \n`{eq_multi}`")

                m = res_multi["metrics"]
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("결정계수 (R²)", f"{m['R2']:.3f}")
                m2.metric("조정된 R² (Adj R²)", f"{m['Adj_R2']:.3f}")
                m3.metric("MAE", f"{m['MAE']:.3f}")
                m4.metric("MSE", f"{m['MSE']:.3f}")
                m5.metric("RMSE", f"{m['RMSE']:.3f}")

                st.markdown("---")
                st.subheader("📊 2. 회귀계수(Coefficient) 분석")

                coef_df = pd.DataFrame(
                    {"독립변수(X)": x_cols_multi, "회귀계수(Beta)": res_multi["coefs"]}
                )

                col_coef_tbl, col_coef_chart = st.columns([1, 1])

                with col_coef_tbl:
                    st.dataframe(coef_df, use_container_width=True)
                    st.warning(
                        "⚠️ **주의:** 다중선형회귀의 회귀계수는 다른 입력 변수들이 일정하다고 가정했을 때 해당 변수가 1만큼 변할 때의 예측값 변화를 의미합니다."
                    )
                    st.info(
                        "💡 변수의 단위가 다르면 계수의 절대 크기만으로 어떤 변수가 더 중요한지 직접 비교하기 어렵습니다."
                    )

                with col_coef_chart:
                    fig_coef = px.bar(
                        coef_df,
                        x="독립변수(X)",
                        y="회귀계수(Beta)",
                        title="독립변수별 회귀계수 크기 비교",
                        color="회귀계수(Beta)",
                        color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)

                st.markdown("---")
                st.subheader("🔮 3. 다중 변수 입력 및 예측")

                st.write("각 독립변수의 값을 설정하여 결과를 예측해보세요:")

                input_data = {}
                cols = st.columns(min(len(x_cols_multi), 4))
                for idx, col in enumerate(x_cols_multi):
                    col_min = float(df[col].min())
                    col_max = float(df[col].max())
                    col_mean = float(df[col].mean())
                    with cols[idx % 4]:
                        input_data[col] = st.number_input(
                            f"`{col}` 값:",
                            min_value=col_min - (col_max - col_min),
                            max_value=col_max + (col_max - col_min),
                            value=col_mean,
                            key=f"input_multi_{col}",
                        )

                input_df = pd.DataFrame([input_data])

                if res_multi["use_scaler"]:
                    pred_multi = res_multi["pipeline"].predict(input_df)[0]
                else:
                    pred_multi = res_multi["model"].predict(input_df)[0]

                st.metric(
                    label=f"🎯 다중선형회귀 예측된 `{y_col_multi}` 값",
                    value=f"{pred_multi:.2f}",
                )

                with st.expander("❓ [탐구 질문 5 & 6 & 7] 다중 회귀 모델의 특성"):
                    st.markdown(
                        """
                    1. **변수를 새로 추가할 때마다 R² 값은 어떻게 변하나요?** 무조건 증가하나요?
                    2. **조정된 R²(Adjusted R²)**는 기존 R²와 어떤 차이점이 있나요?
                    3. **독립변수를 무조건 많이 넣으면** 항상 뛰어난 성능을 가진 좋은 인공지능 모델이 될까요?
                    """
                    )


# ------------------------------------------
# TAB 6: 모델 평가 및 비교
# ------------------------------------------
with tab6:
    st.header("⚖️ 모델 평가 및 비교 (Evaluation)")

    simple_res = st.session_state.get("simple_model_res")
    multi_res = st.session_state.get("multi_model_res")

    if simple_res is None or multi_res is None:
        st.warning(
            "⚠️ 모델 비교를 위해 **[4️⃣ 단순선형회귀]** 탭과 **[5️⃣ 다중선형회귀]** 탭에서 모델을 먼저 학습시켜 주세요."
        )
    elif simple_res["y_col"] != multi_res["y_col"]:
        st.error(
            f"❌ 단순선형회귀의 종속변수(`{simple_res['y_col']}`)와 다중선형회귀의 종속변수(`{multi_res['y_col']}`)가 다릅니다. 동일한 종속변수로 학습 후 비교해주세요."
        )
    else:
        st.subheader("📊 1. 단순 vs 다중 회귀 성능 성능표 비교")

        sm = simple_res["metrics"]
        mm = multi_res["metrics"]

        comp_data = {
            "모델 유형": ["단순선형회귀", "다중선형회귀"],
            "사용한 독립변수": [
                f"{simple_res['x_col']} (1개)",
                f"{', '.join(multi_res['x_cols'])} ({len(multi_res['x_cols'])}개)",
            ],
            "R² (설명력)": [sm["R2"], mm["R2"]],
            "조정된 R²": [sm["Adj_R2"], mm["Adj_R2"]],
            "MAE (절댓값 오차)": [sm["MAE"], mm["MAE"]],
            "MSE (제곱 오차)": [sm["MSE"], mm["MSE"]],
            "RMSE (제곱근 오차)": [sm["RMSE"], mm["RMSE"]],
        }

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["R² (설명력)", "조정된 R²"], color="#lightgreen"
            ).highlight_min(
                subset=["MAE (절댓값 오차)", "MSE (제곱 오차)", "RMSE (제곱근 오차)"],
                color="#lightgreen",
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("📘 지표 개념 가이드")
        g1, g2, g3 = st.columns(3)
        g1.write("**MAE:** 오차 절댓값의 평균 (직관적 해석)")
        g2.write("**RMSE:** 제곱 오차에 루트를 씌워 실제 단위와 맞춘 지표 (큰 오차 가중치)")
        g3.write("**R²:** 모델이 데이터의 변동을 설명하는 비율 (1에 가까울수록 적합)")

        st.markdown("---")
        st.subheader("📈 2. 시각적 잔차 및 예측 성능 비교")

        # 잔차 계산
        simple_residuals = simple_res["y_test"] - simple_res["y_pred"]
        multi_residuals = multi_res["y_test"] - multi_res["y_pred"]

        c_graph1, c_graph2 = st.columns(2)

        with c_graph1:
            st.markdown("##### 🎯 실제값 vs 예측값 산점도")
            fig_act_pred = go.Figure()

            # 단순
            fig_act_pred.add_trace(
                go.Scatter(
                    x=simple_res["y_test"],
                    y=simple_res["y_pred"],
                    mode="markers",
                    name="단순선형회귀",
                    marker=dict(color="blue", opacity=0.6),
                )
            )
            # 다중
            fig_act_pred.add_trace(
                go.Scatter(
                    x=multi_res["y_test"],
                    y=multi_res["y_pred"],
                    mode="markers",
                    name="다중선형회귀",
                    marker=dict(color="green", opacity=0.6, symbol="diamond"),
                )
            )

            # 기준선 (y=x)
            min_v = min(simple_res["y_test"].min(), multi_res["y_test"].min())
            max_v = max(simple_res["y_test"].max(), multi_res["y_test"].max())
            fig_act_pred.add_trace(
                go.Scatter(
                    x=[min_v, max_v],
                    y=[min_v, max_v],
                    mode="lines",
                    name="이상적 기준선 (y=x)",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_act_pred.update_layout(
                xaxis_title="실제값 (Actual)", yaxis_title="예측값 (Predicted)"
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with c_graph2:
            st.markdown("##### 📉 잔차(Residual) 산점도")
            fig_res_scat = go.Figure()

            fig_res_scat.add_trace(
                go.Scatter(
                    x=simple_res["y_pred"],
                    y=simple_residuals,
                    mode="markers",
                    name="단순 회귀 잔차",
                    marker=dict(color="blue", opacity=0.6),
                )
            )
            fig_res_scat.add_trace(
                go.Scatter(
                    x=multi_res["y_pred"],
                    y=multi_residuals,
                    mode="markers",
                    name="다중 회귀 잔차",
                    marker=dict(color="green", opacity=0.6, symbol="diamond"),
                )
            )
            fig_res_scat.add_hline(y=0, line_dash="dash", line_color="red")

            fig_res_scat.update_layout(
                xaxis_title="예측값 (Predicted)", yaxis_title="잔차 (Actual - Predicted)"
            )
            st.plotly_chart(fig_res_scat, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 모델 평가 결과 해석 가이드")
        st.info(
            """
        - **점들이 빨간 기준선(y=x)에 가까울수록** 예측이 실제값과 일치한다는 의미입니다.
        - **잔차가 0을 중심으로 위아래 무작위로 고르게 분포**할 때, 선형회귀 모델의 가정이 잘 충족된 것입니다.
        - **잔차 그래프에서 특정한 곡선이나 패턴이 보이는 경우:** 데이터 간의 관계가 '선형'이 아닐 수 있음을 암시합니다.
        - **판단 기준:** 단순히 R²가 높아졌다고 해서 무조건 다중 모델을 선택하기보다는, 복잡성에 비례하여 오차(RMSE, MAE)가 충분히 개선되었는지 종합적으로 고려해야 합니다.
        """
        )

        with st.expander("❓ [탐구 질문 8 & 9 & 10] 최종 결론 내리기"):
            st.markdown(
                """
            1. **단순회귀와 다중회귀 중 어떤 모델이 데이터의 오차가 더 적은가요?**
            2. **R² 수치가 높아졌을 때 RMSE 도 반드시 항상 함께 감소했나요?**
            3. **상관관계가 높은 독립변수를 발견했다고 해서, 이를 바탕으로 '원인과 결과'라고 확신하여 말할 수 있을까요?**
            4. **이 미세먼지 예측 모델을 실제 내일의 미세먼지 기상 예보 서비스로 바로 활용해도 될까요?** 고려해야 할 한계점은 무엇인가요?
            """
            )
