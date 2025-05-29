"""
HITRAN CRDS 시뮬레이터 - 혼합 가스 분석 + 데이터 내보내기 (Session State)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    page_title="HITRAN CRDS Simulator",
    page_icon="🌟",
    layout="wide"
)

# Session State 초기화
if 'calculation_results' not in st.session_state:
    st.session_state.calculation_results = None
if 'calculation_params' not in st.session_state:
    st.session_state.calculation_params = None

# 제목
st.title("🌟 HITRAN CRDS Spectrum Simulator")
st.markdown("**혼합 가스 스펙트럼 시뮬레이션 도구**")

# 사이드바 - 파라미터 설정
st.sidebar.header("📊 시뮬레이션 파라미터")

# 분자 다중 선택
available_molecules = ["H2O", "CO2", "CH4", "NH3", "N2O", "CO", "O3", "SO2", "NO2", "HNO3"]
selected_molecules = st.sidebar.multiselect(
    "분자 선택 (최대 10개)",
    available_molecules,
    default=["H2O"],
    max_selections=10
)

# 파장 범위
st.sidebar.subheader("파장 범위 (nm)")
col1, col2 = st.sidebar.columns(2)
with col1:
    wavelength_min = st.number_input("최소", value=1500, min_value=100, max_value=10000, step=1)
with col2:
    wavelength_max = st.number_input("최대", value=1520, min_value=100, max_value=10000, step=1)

# 물리 조건
st.sidebar.subheader("물리 조건")

# 온도 (K)
temperature = st.sidebar.number_input(
    "온도 (K)", 
    value=296.15, 
    min_value=200.0, 
    max_value=400.0, 
    step=0.1,
    format="%.2f"
)

# 압력 (torr)
pressure_torr = st.sidebar.number_input(
    "압력 (torr)", 
    value=5320.0,
    min_value=1.0, 
    max_value=15000.0, 
    step=1.0,
    format="%.1f"
)

# 경로 길이 (m)
path_length_m = st.sidebar.number_input(
    "경로 길이 (m)", 
    value=30000.0,
    min_value=1.0, 
    max_value=50000.0, 
    step=1.0,
    format="%.0f"
)

# 분자별 농도 설정
if selected_molecules:
    st.sidebar.subheader("🧪 분자별 농도 (ppb)")
    molecule_concentrations = {}
    
    for molecule in selected_molecules:
        concentration = st.sidebar.number_input(
            f"{molecule} (ppb)",
            value=1000.0 if molecule == "H2O" else 400.0,
            min_value=0.1,
            max_value=10000000.0,
            step=0.1,
            format="%.1f",
            key=f"conc_{molecule}"
        )
        molecule_concentrations[molecule] = concentration

# 계산 버튼
calculate_button = st.sidebar.button("🧮 혼합 스펙트럼 계산", type="primary")

# 결과 초기화 버튼
if st.session_state.calculation_results is not None:
    clear_button = st.sidebar.button("🗑️ 결과 초기화", type="secondary")
    if clear_button:
        st.session_state.calculation_results = None
        st.session_state.calculation_params = None
        st.rerun()

# 단위 변환
pressure_atm = pressure_torr / 760.0
path_length_km = path_length_m / 1000.0

# 분자별 추천 파장 범위 정보
wavelength_ranges = {
    "H2O": "1350-1950nm, 2500-3000nm",
    "CO2": "2000-2100nm, 4200-4400nm",
    "CH4": "1600-1700nm, 3200-3400nm",
    "NH3": "1500-1600nm, 3000-3100nm",
    "N2O": "2200-2300nm, 4400-4600nm",
    "CO": "2300-2400nm, 4600-4800nm",
    "O3": "9600-10000nm",
    "SO2": "7300-7500nm, 8600-8800nm",
    "NO2": "6200-6300nm",
    "HNO3": "1300-1400nm, 7500-7600nm"
}

# 메인 화면
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📋 현재 설정")
    st.write(f"**선택된 분자:** {', '.join(selected_molecules) if selected_molecules else '없음'}")
    st.write(f"**온도:** {temperature} K ({temperature-273.15:.1f}°C)")
    st.write(f"**압력:** {pressure_torr:.1f} torr ({pressure_atm:.2f} atm)")
    st.write(f"**경로 길이:** {path_length_m:.0f} m ({path_length_km:.1f} km)")
    st.write(f"**파장 범위:** {wavelength_min}-{wavelength_max} nm")
    
    # 분자별 농도 표시
    if selected_molecules:
        st.subheader("🧪 분자별 농도")
        for molecule in selected_molecules:
            conc_ppb = molecule_concentrations.get(molecule, 0)
            conc_ppm = conc_ppb / 1000.0
            st.write(f"**{molecule}:** {conc_ppb:.1f} ppb ({conc_ppm:.3f} ppm)")
    
    # 혼합 가스 총 농도
    if selected_molecules:
        total_ppb = sum(molecule_concentrations.values())
        st.write(f"**총 농도:** {total_ppb:.1f} ppb ({total_ppb/1000:.3f} ppm)")
    
    # 추천 파장 범위
    if selected_molecules:
        st.subheader("💡 추천 파장 범위")
        for molecule in selected_molecules:
            st.info(f"**{molecule}:** {wavelength_ranges.get(molecule, '정보 없음')}")

# 계산 실행
with col1:
    if calculate_button and selected_molecules:
        if wavelength_min >= wavelength_max:
            st.error("❌ 최소 파장이 최대 파장보다 작아야 합니다!")
        else:
            # 진행 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 주파수 격자 생성
                freq_min = 1e7 / wavelength_max
                freq_max = 1e7 / wavelength_min
                frequency_grid = np.linspace(freq_min, freq_max, 5000)
                
                # 각 분자별 스펙트럼 계산
                hitran_api = HitranAPI()
                calc = SpectrumCalculator()
                
                individual_spectra = {}
                combined_absorption = np.zeros_like(frequency_grid)
                
                for i, molecule in enumerate(selected_molecules):
                    progress = int(20 + (i / len(selected_molecules)) * 60)
                    status_text.text(f"📥 {molecule} 데이터 다운로드 중... ({i+1}/{len(selected_molecules)})")
                    progress_bar.progress(progress)
                    
                    # HITRAN 데이터 다운로드
                    hitran_data = hitran_api.download_molecule_data(molecule, wavelength_min, wavelength_max)
                    
                    if hitran_data is not None and len(hitran_data) > 0:
                        # 개별 스펙트럼 계산
                        concentration = molecule_concentrations[molecule] / 1e9  # ppb to 몰분율
                        
                        spectrum = calc.calculate_absorption_spectrum(
                            hitran_data=hitran_data,
                            frequency_grid=frequency_grid,
                            temperature=temperature,
                            pressure=pressure_atm,
                            concentration=concentration,
                            path_length=path_length_m,
                            molecule=molecule
                        )
                        
                        individual_spectra[molecule] = spectrum
                        combined_absorption += spectrum['absorption_coeff']
                    else:
                        st.warning(f"⚠️ {molecule} 데이터를 찾을 수 없습니다 (파장 범위: {wavelength_min}-{wavelength_max}nm)")
                
                # 혼합 스펙트럼 계산
                status_text.text("🧮 혼합 스펙트럼 계산 중...")
                progress_bar.progress(80)
                
                combined_transmittance = np.exp(-combined_absorption * path_length_m)
                combined_absorbance = -np.log10(combined_transmittance)
                wavelength_nm = 1e7 / frequency_grid
                
                # 분자별 기여도 계산
                contribution_data = []
                for molecule, spectrum in individual_spectra.items():
                    max_abs = np.max(spectrum['absorbance'])
                    contribution_data.append({
                        '분자': molecule,
                        '최대 흡광도': f"{max_abs:.4f}",
                        '농도 (ppb)': f"{molecule_concentrations[molecule]:.1f}",
                        '기여율': f"{(max_abs / np.max(combined_absorbance) * 100):.1f}%" if np.max(combined_absorbance) > 0 else "0%"
                    })
                
                # Session State에 결과 저장
                st.session_state.calculation_results = {
                    'individual_spectra': individual_spectra,
                    'combined_transmittance': combined_transmittance,
                    'combined_absorbance': combined_absorbance,
                    'wavelength_nm': wavelength_nm,
                    'combined_absorption': combined_absorption,
                    'contribution_data': contribution_data,
                    'total_lines': sum(len(hitran_api.download_molecule_data(mol, wavelength_min, wavelength_max) or []) for mol in selected_molecules)
                }
                
                # 파라미터도 저장
                st.session_state.calculation_params = {
                    'selected_molecules': selected_molecules.copy(),
                    'wavelength_min': wavelength_min,
                    'wavelength_max': wavelength_max,
                    'temperature': temperature,
                    'pressure_torr': pressure_torr,
                    'pressure_atm': pressure_atm,
                    'path_length_m': path_length_m,
                    'path_length_km': path_length_km,
                    'molecule_concentrations': molecule_concentrations.copy()
                }
                
                progress_bar.progress(100)
                status_text.text("✅ 혼합 스펙트럼 계산 완료!")
                
            except Exception as e:
                st.error(f"❌ 에러 발생: {str(e)}")
                progress_bar.empty()
                status_text.empty()
    
    elif calculate_button and not selected_molecules:
        st.warning("⚠️ 분석할 분자를 선택해주세요!")

# 저장된 결과 표시
if st.session_state.calculation_results is not None:
    results = st.session_state.calculation_results
    params = st.session_state.calculation_params
    
    with col1:
        # 그래프 생성
        st.subheader("📊 스펙트럼 결과")
        
        # 색상 팔레트
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Plotly 그래프 생성
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('개별 분자 흡광도', '혼합 투과율', '혼합 흡광도'),
            vertical_spacing=0.08
        )
        
        # 1. 개별 분자 흡광도
        for i, (molecule, spectrum) in enumerate(results['individual_spectra'].items()):
            fig.add_trace(
                go.Scatter(
                    x=results['wavelength_nm'],
                    y=spectrum['absorbance'],
                    mode='lines',
                    name=f'{molecule}',
                    line=dict(color=colors[i % len(colors)], width=1)
                ),
                row=1, col=1
            )
        
        # 2. 혼합 투과율
        fig.add_trace(
            go.Scatter(
                x=results['wavelength_nm'],
                y=results['combined_transmittance'],
                mode='lines',
                name='혼합 투과율',
                line=dict(color='black', width=2)
            ),
            row=2, col=1
        )
        
        # 3. 혼합 흡광도
        fig.add_trace(
            go.Scatter(
                x=results['wavelength_nm'],
                y=results['combined_absorbance'],
                mode='lines',
                name='혼합 흡광도',
                line=dict(color='darkred', width=2)
            ),
            row=3, col=1
        )
        
        # 레이아웃 설정
        fig.update_layout(
            height=900,
            title=f"혼합 가스 스펙트럼 ({', '.join(params['selected_molecules'])})",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="파장 (nm)", row=3, col=1)
        fig.update_yaxes(title_text="흡광도", row=1, col=1)
        fig.update_yaxes(title_text="투과율", row=2, col=1)
        fig.update_yaxes(title_text="흡광도", row=3, col=1)
        
        # 그래프 표시
        st.plotly_chart(fig, use_container_width=True)
        
        # 결과 분석
        st.subheader("📈 분석 결과")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("총 HITRAN 라인 수", f"{results['total_lines']:,}")
        
        with col_b:
            st.metric("최소 투과율", f"{np.min(results['combined_transmittance']):.4f}")
        
        with col_c:
            st.metric("최대 흡광도", f"{np.max(results['combined_absorbance']):.4f}")
        
        # 분자별 기여도
        st.subheader("🔍 분자별 기여도")
        df = pd.DataFrame(results['contribution_data'])
        st.dataframe(df, use_container_width=True)
        
        # 스펙트럼 간섭 분석
        if len(results['individual_spectra']) > 1:
            st.subheader("⚠️ 스펙트럼 간섭 분석")
            overlap_threshold = 0.01
            
            overlap_regions = []
            for i in range(len(results['wavelength_nm'])):
                overlapping_molecules = []
                for molecule, spectrum in results['individual_spectra'].items():
                    if spectrum['absorbance'][i] > overlap_threshold:
                        overlapping_molecules.append(molecule)
                
                if len(overlapping_molecules) > 1:
                    overlap_regions.append({
                        '파장 (nm)': f"{results['wavelength_nm'][i]:.1f}",
                        '간섭 분자': ', '.join(overlapping_molecules),
                        '간섭 강도': 'High' if len(overlapping_molecules) > 2 else 'Medium'
                    })
            
            if overlap_regions:
                unique_overlaps = {}
                for region in overlap_regions:
                    key = region['간섭 분자']
                    if key not in unique_overlaps:
                        unique_overlaps[key] = region
                
                st.warning(f"🔍 {len(unique_overlaps)}개의 스펙트럼 간섭 영역이 발견되었습니다.")
                overlap_df = pd.DataFrame(list(unique_overlaps.values()))
                st.dataframe(overlap_df, use_container_width=True)
            else:
                st.success("✅ 선택한 파장 범위에서 심각한 스펙트럼 간섭이 발견되지 않았습니다.")
        
        # 데이터 내보내기 섹션
        st.subheader("📁 데이터 내보내기")
        
        col_download1, col_download2, col_download3, col_download4 = st.columns(4)
        
        with col_download1:
            # 스펙트럼 데이터 CSV
            spectrum_data = {
                'Wavelength_nm': results['wavelength_nm'],
                'Combined_Transmittance': results['combined_transmittance'],
                'Combined_Absorbance': results['combined_absorbance']
            }
            
            # 개별 분자 데이터 추가
            for molecule, spectrum in results['individual_spectra'].items():
                spectrum_data[f'{molecule}_Transmittance'] = spectrum['transmittance']
                spectrum_data[f'{molecule}_Absorbance'] = spectrum['absorbance']
            
            spectrum_df = pd.DataFrame(spectrum_data)
            csv_data = spectrum_df.to_csv(index=False)
            
            st.download_button(
                label="📊 스펙트럼 데이터 (CSV)",
                data=csv_data,
                file_name=f"spectrum_data_{'-'.join(params['selected_molecules'])}_{params['wavelength_min']}-{params['wavelength_max']}nm.csv",
                mime="text/csv"
            )
        
        with col_download2:
            # 계산 조건 요약
            summary_text = f"""HITRAN CRDS 시뮬레이션 결과 요약
=================================

계산 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

분석 조건:
- 선택 분자: {', '.join(params['selected_molecules'])}
- 온도: {params['temperature']} K ({params['temperature']-273.15:.1f}°C)
- 압력: {params['pressure_torr']:.1f} torr ({params['pressure_atm']:.2f} atm)
- 경로 길이: {params['path_length_m']:.0f} m ({params['path_length_km']:.1f} km)
- 파장 범위: {params['wavelength_min']}-{params['wavelength_max']} nm

분자별 농도:
"""
            for molecule in params['selected_molecules']:
                conc_ppb = params['molecule_concentrations'].get(molecule, 0)
                summary_text += f"- {molecule}: {conc_ppb:.1f} ppb ({conc_ppb/1000:.3f} ppm)\n"
            
            summary_text += f"\n총 농도: {sum(params['molecule_concentrations'].values()):.1f} ppb\n"
            
            summary_text += f"""
분석 결과:
- 최소 투과율: {np.min(results['combined_transmittance']):.4f}
- 최대 흡광도: {np.max(results['combined_absorbance']):.4f}
- 총 HITRAN 라인 수: {results['total_lines']:,}

분자별 기여도:
"""
            for data in results['contribution_data']:
                summary_text += f"- {data['분자']}: 최대 흡광도 {data['최대 흡광도']}, 기여율 {data['기여율']}\n"
            
            st.download_button(
                label="📋 분석 요약 (TXT)",
                data=summary_text,
                file_name=f"analysis_summary_{'-'.join(params['selected_molecules'])}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col_download3:
            # 기여도 데이터 엑셀
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # 기본 정보
                info_df = pd.DataFrame({
                    '항목': ['온도 (K)', '압력 (torr)', '경로길이 (m)', '파장범위 (nm)', '총농도 (ppb)'],
                    '값': [params['temperature'], params['pressure_torr'], params['path_length_m'], f"{params['wavelength_min']}-{params['wavelength_max']}", sum(params['molecule_concentrations'].values())]
                })
                info_df.to_excel(writer, sheet_name='분석조건', index=False)
                
                # 기여도 데이터
                df.to_excel(writer, sheet_name='분자별기여도', index=False)
                
                # 스펙트럼 데이터 (샘플링)
                sample_spectrum = spectrum_df.iloc[::10]
                sample_spectrum.to_excel(writer, sheet_name='스펙트럼데이터', index=False)
            
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📊 분석 데이터 (Excel)",
                data=excel_data,
                file_name=f"crds_analysis_{'-'.join(params['selected_molecules'])}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col_download4:
            st.info("🖼️ 그래프 이미지는 그래프 우상단의 📷 버튼을 클릭하여 다운로드할 수 있습니다!")

# 하단 정보
st.markdown("---")
st.markdown("**개발:** HITRAN CRDS Simulator v2.1 (Session State + 데이터 내보내기) | **데이터:** HITRAN Database")
st.markdown("**지원 분자:** H2O, CO2, CH4, NH3, N2O, CO, O3, SO2, NO2, HNO3")