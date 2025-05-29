"""
HITRAN CRDS 대기 프로파일 시뮬레이션
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import io
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_handler.hitran_api import HitranAPI
from spectrum_calc.absorption import SpectrumCalculator

# 페이지 설정
st.set_page_config(
    page_title="HITRAN Atmospheric Profile",
    page_icon="🌍",
    layout="wide"
)

# Session State 초기화
if 'atmosphere_results' not in st.session_state:
    st.session_state.atmosphere_results = None

st.title("🌍 HITRAN CRDS Atmospheric Profile Simulator")
st.markdown("**실제 대기 조건을 고려한 스펙트럼 시뮬레이션**")

# 대기 모델 함수들
def us_standard_atmosphere(altitude_km):
    """
    US Standard Atmosphere (1976) 모델
    
    Args:
        altitude_km: 고도 (km)
    
    Returns:
        temperature (K), pressure (Pa), density (kg/m3)
    """
    h = altitude_km * 1000  # m 단위로 변환
    
    # 대기층별 매개변수
    if h <= 11000:  # 대류권
        T = 288.15 - 0.0065 * h
        P = 101325 * (T / 288.15) ** 5.2561
    elif h <= 20000:  # 하부 성층권
        T = 216.65
        P = 22632 * np.exp(-0.00015769 * (h - 11000))
    elif h <= 32000:  # 상부 성층권
        T = 216.65 + 0.001 * (h - 20000)
        P = 5474.9 * (T / 216.65) ** (-34.163)
    elif h <= 47000:  # 중간권 하부
        T = 228.65 + 0.0028 * (h - 32000)
        P = 868.02 * (T / 228.65) ** (-12.201)
    elif h <= 51000:  # 중간권 상부
        T = 270.65
        P = 110.91 * np.exp(-0.00012622 * (h - 47000))
    else:  # 열권
        T = 270.65 - 0.0028 * (h - 51000)
        P = 66.939 * (T / 270.65) ** 12.201
    
    # 밀도 계산 (이상기체 법칙)
    R = 287.0  # 기체상수 (J/kg/K)
    rho = P / (R * T)
    
    return T, P, rho

def tropical_atmosphere(altitude_km):
    """열대 대기 모델"""
    T_std, P_std, rho_std = us_standard_atmosphere(altitude_km)
    
    # 열대 지역 보정
    if altitude_km <= 10:
        T = T_std + 5  # 지표면 더 따뜻
        P = P_std * 1.05  # 압력 약간 높음
    else:
        T = T_std + 2
        P = P_std * 0.98
    
    rho = P / (287.0 * T)
    return T, P, rho

def polar_atmosphere(altitude_km):
    """극지 대기 모델"""
    T_std, P_std, rho_std = us_standard_atmosphere(altitude_km)
    
    # 극지 보정
    if altitude_km <= 10:
        T = T_std - 20  # 지표면 더 추움
        P = P_std * 0.95  # 압력 약간 낮음
    else:
        T = T_std - 10
        P = P_std * 0.92
    
    rho = P / (287.0 * T)
    return T, P, rho

# 사이드바 설정
st.sidebar.header("🌍 대기 조건 설정")

# 대기 모델 선택
atmosphere_model = st.sidebar.selectbox(
    "대기 모델 선택",
    ["US Standard (1976)", "Tropical", "Polar", "Custom"]
)

# 고도 범위
st.sidebar.subheader("📏 관측 경로 설정")
observation_type = st.sidebar.selectbox(
    "관측 타입",
    ["지상 관측", "항공기 관측", "위성 관측", "사용자 정의"]
)

if observation_type == "지상 관측":
    altitude_start = 0.0
    altitude_end = st.sidebar.slider("관측 고도 (km)", 0.1, 10.0, 5.0, 0.1)
    path_type = "수평"
elif observation_type == "항공기 관측":
    altitude_start = st.sidebar.slider("시작 고도 (km)", 0.0, 15.0, 0.0, 0.1)
    altitude_end = st.sidebar.slider("끝 고도 (km)", 0.0, 15.0, 10.0, 0.1)
    path_type = "경사"
elif observation_type == "위성 관측":
    altitude_start = st.sidebar.slider("시작 고도 (km)", 0.0, 50.0, 0.0, 0.5)
    altitude_end = st.sidebar.slider("끝 고도 (km)", 0.0, 50.0, 30.0, 0.5)
    path_type = "수직"
else:
    altitude_start = st.sidebar.number_input("시작 고도 (km)", 0.0, 50.0, 0.0, 0.1)
    altitude_end = st.sidebar.number_input("끝 고도 (km)", 0.0, 50.0, 10.0, 0.1)
    path_type = st.sidebar.selectbox("경로 타입", ["수평", "경사", "수직"])

# 분자 및 파장 설정
st.sidebar.subheader("🧪 분자 설정")
molecule = st.sidebar.selectbox("분자 선택", ["H2O", "CO2", "CH4", "NH3", "N2O"])

col1, col2 = st.sidebar.columns(2)
with col1:
    wavelength_min = st.number_input("최소 파장 (nm)", value=1500, min_value=100, max_value=10000)
with col2:
    wavelength_max = st.number_input("최대 파장 (nm)", value=1520, min_value=100, max_value=10000)

# 농도 프로파일
st.sidebar.subheader("🌫️ 농도 프로파일")
concentration_type = st.sidebar.selectbox(
    "농도 분포",
    ["균일 분포", "지수 감소", "실제 프로파일"]
)

if concentration_type == "균일 분포":
    base_concentration = st.sidebar.number_input("농도 (ppb)", 100.0, 10000.0, 1000.0)
elif concentration_type == "지수 감소":
    surface_concentration = st.sidebar.number_input("지표 농도 (ppb)", 100.0, 10000.0, 1000.0)
    scale_height = st.sidebar.number_input("스케일 높이 (km)", 1.0, 20.0, 8.0)
else:  # 실제 프로파일
    if molecule == "H2O":
        st.sidebar.info("💧 H2O: 지표면 높음, 고도에 따라 급격히 감소")
    elif molecule == "CO2":
        st.sidebar.info("🌍 CO2: 비교적 균일, 약간의 고도 의존성")
    elif molecule == "CH4":
        st.sidebar.info("🔥 CH4: 지표면 높음, 성층권에서 감소")

# 계산 버튼
calculate_button = st.sidebar.button("🧮 대기 프로파일 계산", type="primary")

# 결과 초기화
if st.session_state.atmosphere_results:
    if st.sidebar.button("🗑️ 결과 초기화"):
        st.session_state.atmosphere_results = None
        st.rerun()

# 메인 화면
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📋 현재 설정")
    st.write(f"**대기 모델:** {atmosphere_model}")
    st.write(f"**관측 타입:** {observation_type}")
    st.write(f"**고도 범위:** {altitude_start:.1f} - {altitude_end:.1f} km")
    st.write(f"**경로 타입:** {path_type}")
    st.write(f"**분자:** {molecule}")
    st.write(f"**파장 범위:** {wavelength_min}-{wavelength_max} nm")
    st.write(f"**농도 분포:** {concentration_type}")

# 계산 실행
with col1:
    if calculate_button:
        if wavelength_min >= wavelength_max:
            st.error("❌ 최소 파장이 최대 파장보다 작아야 합니다!")
        elif altitude_start >= altitude_end:
            st.error("❌ 시작 고도가 끝 고도보다 작아야 합니다!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. 대기 프로파일 생성
                status_text.text("🌍 대기 프로파일 생성 중...")
                progress_bar.progress(20)
                
                # 고도 격자 생성
                num_layers = 50
                altitudes = np.linspace(altitude_start, altitude_end, num_layers)
                
                # 대기 모델별 프로파일 계산
                temperatures = []
                pressures = []
                densities = []
                
                for alt in altitudes:
                    if atmosphere_model == "US Standard (1976)":
                        T, P, rho = us_standard_atmosphere(alt)
                    elif atmosphere_model == "Tropical":
                        T, P, rho = tropical_atmosphere(alt)
                    elif atmosphere_model == "Polar":
                        T, P, rho = polar_atmosphere(alt)
                    else:  # Custom
                        T, P, rho = us_standard_atmosphere(alt)  # 기본값
                    
                    temperatures.append(T)
                    pressures.append(P / 100)  # Pa to hPa
                    densities.append(rho)
                
                # 2. 농도 프로파일 생성
                status_text.text("🌫️ 농도 프로파일 생성 중...")
                progress_bar.progress(40)
                
                concentrations = []
                for alt in altitudes:
                    if concentration_type == "균일 분포":
                        conc = base_concentration
                    elif concentration_type == "지수 감소":
                        conc = surface_concentration * np.exp(-alt / scale_height)
                    else:  # 실제 프로파일
                        if molecule == "H2O":
                            conc = 10000 * np.exp(-alt / 2.0)  # 수증기는 빠르게 감소
                        elif molecule == "CO2":
                            conc = 400000 * (1 - 0.1 * alt / 50)  # 약간 감소
                        elif molecule == "CH4":
                            conc = 1800 * np.exp(-alt / 8.0)  # 메탄은 천천히 감소
                        else:
                            conc = 1000 * np.exp(-alt / 5.0)  # 기본값
                    
                    concentrations.append(max(conc, 0.1))  # 최소값 제한
                
                # 3. HITRAN 데이터 다운로드
                status_text.text("📥 HITRAN 데이터 다운로드 중...")
                progress_bar.progress(60)
                
                hitran_api = HitranAPI()
                hitran_data = hitran_api.download_molecule_data(molecule, wavelength_min, wavelength_max)
                
                if hitran_data is None or len(hitran_data) == 0:
                    st.error(f"❌ {molecule} 데이터를 찾을 수 없습니다!")
                else:
                    # 4. 층별 스펙트럼 계산
                    status_text.text("🧮 층별 스펙트럼 계산 중...")
                    
                    # 주파수 격자
                    freq_min = 1e7 / wavelength_max
                    freq_max = 1e7 / wavelength_min
                    frequency_grid = np.linspace(freq_min, freq_max, 2000)  # 해상도 낮춤 (속도 향상)
                    wavelength_nm = 1e7 / frequency_grid
                    
                    calc = SpectrumCalculator()
                    
                    # 층별 계산
                    layer_spectra = []
                    total_absorption = np.zeros_like(frequency_grid)
                    
                    layer_thickness = 1000  # 1km 두께
                    
                    for i, (alt, T, P_hPa, conc) in enumerate(zip(altitudes, temperatures, pressures, concentrations)):
                        if i % 10 == 0:
                            progress = 60 + int((i / len(altitudes)) * 30)
                            progress_bar.progress(progress)
                        
                        # 층별 스펙트럼 계산
                        spectrum = calc.calculate_absorption_spectrum(
                            hitran_data=hitran_data,
                            frequency_grid=frequency_grid,
                            temperature=T,
                            pressure=P_hPa / 760.0,  # hPa to atm
                            concentration=conc / 1e9,  # ppb to 몰분율
                            path_length=layer_thickness,  # 1km 층 두께
                            molecule=molecule
                        )
                        
                        layer_spectra.append({
                            'altitude': alt,
                            'temperature': T,
                            'pressure': P_hPa,
                            'concentration': conc,
                            'spectrum': spectrum
                        })
                        
                        # 총 흡수에 누적
                        total_absorption += spectrum['absorption_coeff'] * layer_thickness
                    
                    # 5. 전체 투과율 계산
                    status_text.text("📊 전체 투과율 계산 중...")
                    progress_bar.progress(90)
                    
                    total_transmittance = np.exp(-total_absorption)
                    total_absorbance = -np.log10(total_transmittance)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 대기 프로파일 계산 완료!")
                    
                    # 결과 저장
                    st.session_state.atmosphere_results = {
                        'altitudes': altitudes,
                        'temperatures': temperatures,
                        'pressures': pressures,
                        'concentrations': concentrations,
                        'wavelength_nm': wavelength_nm,
                        'layer_spectra': layer_spectra,
                        'total_transmittance': total_transmittance,
                        'total_absorbance': total_absorbance,
                        'settings': {
                            'atmosphere_model': atmosphere_model,
                            'observation_type': observation_type,
                            'molecule': molecule,
                            'wavelength_range': f"{wavelength_min}-{wavelength_max}nm",
                            'concentration_type': concentration_type
                        }
                    }
                
            except Exception as e:
                st.error(f"❌ 에러 발생: {str(e)}")
                progress_bar.empty()
                status_text.empty()

# 결과 표시
if st.session_state.atmosphere_results:
    results = st.session_state.atmosphere_results
    
    with col1:
        st.subheader("🌍 대기 프로파일 결과")
        
        # 4분할 그래프
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('대기 프로파일', '농도 프로파일', '층별 기여도', '전체 스펙트럼'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 대기 프로파일 (온도, 압력)
        fig.add_trace(
            go.Scatter(
                x=results['temperatures'],
                y=results['altitudes'],
                mode='lines',
                name='온도 (K)',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=results['pressures'],
                y=results['altitudes'],
                mode='lines',
                name='압력 (hPa)',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. 농도 프로파일
        fig.add_trace(
            go.Scatter(
                x=results['concentrations'],
                y=results['altitudes'],
                mode='lines',
                name=f'{results["settings"]["molecule"]} (ppb)',
                line=dict(color='green', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. 층별 기여도 (최대 흡광도)
        layer_contributions = [np.max(layer['spectrum']['absorbance']) for layer in results['layer_spectra']]
        
        fig.add_trace(
            go.Scatter(
                x=layer_contributions,
                y=results['altitudes'],
                mode='lines+markers',
                name='층별 기여도',
                line=dict(color='orange', width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 전체 스펙트럼
        fig.add_trace(
            go.Scatter(
                x=results['wavelength_nm'],
                y=results['total_transmittance'],
                mode='lines',
                name='투과율',
                line=dict(color='purple', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 레이아웃 설정
        fig.update_layout(height=800, showlegend=True)
        
        # x축 제목
        fig.update_xaxes(title_text="온도 (K)", row=1, col=1)
        fig.update_xaxes(title_text=f"{results['settings']['molecule']} (ppb)", row=1, col=2)
        fig.update_xaxes(title_text="최대 흡광도", row=2, col=1)
        fig.update_xaxes(title_text="파장 (nm)", row=2, col=2)
        
        # y축 제목
        fig.update_yaxes(title_text="고도 (km)", row=1, col=1)
        fig.update_yaxes(title_text="고도 (km)", row=1, col=2)
        fig.update_yaxes(title_text="고도 (km)", row=2, col=1)
        fig.update_yaxes(title_text="투과율", row=2, col=2)
        
        # 이차 y축 제목
        fig.update_yaxes(title_text="압력 (hPa)", row=1, col=1, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 통계 정보
        st.subheader("📊 대기 통계")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("평균 온도", f"{np.mean(results['temperatures']):.1f} K")
        
        with col_stat2:
            st.metric("평균 압력", f"{np.mean(results['pressures']):.1f} hPa")
        
        with col_stat3:
            st.metric("총 농도", f"{np.sum(results['concentrations']):.0f} ppb·km")
        
        with col_stat4:
            st.metric("최소 투과율", f"{np.min(results['total_transmittance']):.4f}")
        
        # 층별 상세 정보
        st.subheader("📋 층별 상세 정보")
        
        layer_summary = []
        for i, layer in enumerate(results['layer_spectra'][::5]):  # 5개마다 표시
            layer_summary.append({
                '고도 (km)': f"{layer['altitude']:.1f}",
                '온도 (K)': f"{layer['temperature']:.1f}",
                '압력 (hPa)': f"{layer['pressure']:.1f}",
                '농도 (ppb)': f"{layer['concentration']:.1f}",
                '최대 흡광도': f"{np.max(layer['spectrum']['absorbance']):.4f}"
            })
        
        summary_df = pd.DataFrame(layer_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # 데이터 내보내기
        st.subheader("📁 대기 프로파일 데이터 내보내기")
        
        col_down1, col_down2 = st.columns(2)
        
        with col_down1:
            # 대기 프로파일 CSV
            profile_data = {
                'Altitude_km': results['altitudes'],
                'Temperature_K': results['temperatures'],
                'Pressure_hPa': results['pressures'],
                f'{results["settings"]["molecule"]}_ppb': results['concentrations'],
                'Layer_Contribution': layer_contributions
            }
            
            profile_df = pd.DataFrame(profile_data)
            profile_csv = profile_df.to_csv(index=False)
            
            st.download_button(
                label="🌍 대기 프로파일 (CSV)",
                data=profile_csv,
                file_name=f"atmosphere_profile_{results['settings']['atmosphere_model']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col_down2:
            # 전체 스펙트럼 CSV
            spectrum_data = {
                'Wavelength_nm': results['wavelength_nm'],
                'Transmittance': results['total_transmittance'],
                'Absorbance': results['total_absorbance']
            }
            
            spectrum_df = pd.DataFrame(spectrum_data)
            spectrum_csv = spectrum_df.to_csv(index=False)
            
            st.download_button(
                label="📊 전체 스펙트럼 (CSV)",
                data=spectrum_csv,
                file_name=f"atmosphere_spectrum_{results['settings']['molecule']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# 하단 정보
st.markdown("---")
st.markdown("**개발:** HITRAN CRDS Atmospheric Profile Simulator v1.0 | **데이터:** HITRAN Database + US Standard Atmosphere")
st.markdown("**대기 모델:** US Standard (1976), Tropical, Polar")