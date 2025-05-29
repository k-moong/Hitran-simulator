"""
스펙트럼 흡수 계산 모듈
"""

import numpy as np
import pandas as pd
from scipy.special import wofz
import matplotlib.pyplot as plt

class SpectrumCalculator:
    def __init__(self):
        """스펙트럼 계산기 초기화"""
        # 물리 상수들
        self.c = 2.99792458e8  # 빛의 속도 m/s
        self.k_B = 1.380649e-23  # 볼츠만 상수 J/K
        self.N_A = 6.02214076e23  # 아보가드로 수
        
    def voigt_profile(self, frequency, center_freq, gamma_lorentz, gamma_doppler):
        """
        Voigt 프로파일 계산 (Lorentz + Doppler 혼합)
        
        Args:
            frequency: 주파수 배열 (cm^-1)
            center_freq: 중심 주파수 (cm^-1)
            gamma_lorentz: Lorentz 반폭 (cm^-1)
            gamma_doppler: Doppler 반폭 (cm^-1)
        """
        # 정규화된 주파수 차이
        x = (frequency - center_freq) / gamma_doppler
        y = gamma_lorentz / gamma_doppler
        
        # Voigt 함수 계산 (복소 오차 함수 사용)
        z = x + 1j * y
        w = wofz(z)
        
        # 정규화
        profile = w.real / (gamma_doppler * np.sqrt(np.pi))
        return profile
    
    def calculate_doppler_width(self, center_freq, temperature, molecular_mass):
        """
        도플러 폭 계산
        
        Args:
            center_freq: 중심 주파수 (cm^-1)
            temperature: 온도 (K)
            molecular_mass: 분자량 (g/mol)
        """
        # 도플러 폭 공식
        gamma_doppler = (center_freq / self.c) * np.sqrt(
            2 * self.k_B * temperature * self.N_A / (molecular_mass * 1e-3)
        )
        return gamma_doppler
    
    def calculate_lorentz_width(self, pressure_broadening, pressure, temperature, ref_temp=296.0):
        """
        로렌츠 폭 계산 (압력 확장)
        
        Args:
            pressure_broadening: 압력 확장 계수 (cm^-1/atm)
            pressure: 압력 (atm)
            temperature: 온도 (K)
            ref_temp: 참조 온도 (K)
        """
        # 온도 의존성을 고려한 로렌츠 폭
        gamma_lorentz = pressure_broadening * pressure * (ref_temp / temperature)**0.5
        return gamma_lorentz
    
    def calculate_absorption_spectrum(self, hitran_data, frequency_grid, 
                                   temperature=296.15, pressure=1.0, 
                                   concentration=1000e-6, path_length=1000.0, molecule="H2O"):
        """
        흡수 스펙트럼 계산
        
        Args:
            hitran_data: HITRAN 데이터 (astroquery 결과)
            frequency_grid: 주파수 격자 (cm^-1)
            temperature: 온도 (K)
            pressure: 압력 (atm)
            concentration: 농도 (몰 분율)
            path_length: 경로 길이 (m)
            molecule: 분자 이름
        """
        print(f"🧮 {molecule} 스펙트럼 계산 중...")
        print(f"   온도: {temperature} K")
        print(f"   압력: {pressure} atm")
        print(f"   농도: {concentration*1e6:.1f} ppm")
        print(f"   경로 길이: {path_length/1000:.1f} km")
        
        # 전체 흡수 계수 초기화
        absorption_coeff = np.zeros_like(frequency_grid)
        
        # 분자별 분자량 (g/mol)
        molecular_masses = {
            "H2O": 18.015,
            "CO2": 44.01,
            "CH4": 16.04,
            "NH3": 17.03,
            "N2O": 44.01,
            "CO": 28.01,
            "O3": 47.998,
            "SO2": 64.066,
            "NO2": 46.006,
            "HNO3": 63.01,
            "O2": 31.998,
            "NO": 30.006,
            "OH": 17.007,
            "HF": 20.006,
            "HCl": 36.458,
            "HBr": 80.912,
            "HI": 127.912,
            "ClO": 51.452,
            "OCS": 60.076,
            "H2CO": 30.026,
            "HOCl": 52.460,
            "N2": 28.014,
            "HCN": 27.026,
            "CH3Cl": 50.487,
            "H2O2": 34.015,
            "C2H2": 26.037,
            "C2H6": 30.069,
            "PH3": 33.998
        }
        
        molecular_mass = molecular_masses.get(molecule, 18.015)
        print(f"   분자량: {molecular_mass} g/mol")
        
        # 각 HITRAN 라인에 대해 계산
        for i, line in enumerate(hitran_data):
            if i % 1000 == 0:
                print(f"   진행: {i}/{len(hitran_data)} 라인")
            
            # 라인 파라미터들
            center_freq = line['nu']  # 중심 주파수 (cm^-1)
            intensity = line['sw']    # 선 강도
            gamma_air = line['gamma_air']  # 공기 확장 계수
            
            # 도플러 폭 계산
            gamma_d = self.calculate_doppler_width(center_freq, temperature, molecular_mass)
            
            # 로렌츠 폭 계산
            gamma_l = self.calculate_lorentz_width(gamma_air, pressure, temperature)
            
            # Voigt 프로파일 계산
            line_shape = self.voigt_profile(frequency_grid, center_freq, gamma_l, gamma_d)
            
            # 흡수 계수에 기여 추가 (스케일링 팩터 적용)
            absorption_coeff += intensity * concentration * line_shape * 1e20
        
        # Beer-Lambert 법칙: I = I0 * exp(-alpha * L)
        transmittance = np.exp(-absorption_coeff * path_length)
        absorbance = -np.log10(transmittance)
        
        print(f"✅ {molecule} 스펙트럼 계산 완료!")
        
        return {
            'frequency': frequency_grid,
            'absorption_coeff': absorption_coeff,
            'transmittance': transmittance,
            'absorbance': absorbance
        }

# 테스트 실행
if __name__ == "__main__":
    print("=== 스펙트럼 계산 테스트 ===")
    calc = SpectrumCalculator()
    
    # 샘플 주파수 격자 생성
    freq_min, freq_max = 6250, 6300  # cm^-1
    frequency_grid = np.linspace(freq_min, freq_max, 5000)
    
    print(f"주파수 범위: {freq_min}-{freq_max} cm^-1")
    print(f"해상도: {(freq_max-freq_min)/len(frequency_grid):.6f} cm^-1")
    print("실제 HITRAN 데이터가 필요합니다...")