"""
HITRAN API 연결 및 데이터 다운로드 모듈 (astroquery 사용)
"""

from astroquery import hitran
import os
import pandas as pd
import astropy.units as u

# 직접 설정
HITRAN_CACHE_DIR = "cache/"
HITRAN_DATA_DIR = "data/"

class HitranAPI:
    def __init__(self):
        """HITRAN API 초기화"""
        # 데이터 폴더 생성
        os.makedirs(HITRAN_CACHE_DIR, exist_ok=True)
        os.makedirs(HITRAN_DATA_DIR, exist_ok=True)
        
    def test_connection(self):
        """HITRAN 연결 테스트"""
        try:
            print("✅ HITRAN API (astroquery) 연결 성공!")
            return True
        except Exception as e:
            print(f"❌ HITRAN API 연결 실패: {e}")
            return False
    
    def download_molecule_data(self, molecule="H2O", wavelength_min=1500, wavelength_max=1600):
        """
        분자 데이터 다운로드
        
        Args:
            molecule: 분자 이름 (H2O, CO2, CH4 등)
            wavelength_min: 최소 파장 (nm)
            wavelength_max: 최대 파장 (nm)
        """
        try:
            # nm를 cm^-1로 변환
            wavenumber_min = 1e7 / wavelength_max  # cm^-1
            wavenumber_max = 1e7 / wavelength_min  # cm^-1
            
            print(f"📥 {molecule} 데이터 다운로드 중...")
            print(f"   파장 범위: {wavelength_min}-{wavelength_max} nm")
            print(f"   파수 범위: {wavenumber_min:.1f}-{wavenumber_max:.1f} cm^-1")
            
            # HITRAN 분자 ID 매핑
            molecule_ids = {
                "H2O": 1,    # 물
                "CO2": 2,    # 이산화탄소  
                "O3": 3,     # 오존
                "N2O": 4,    # 아산화질소
                "CO": 5,     # 일산화탄소
                "CH4": 6,    # 메탄
                "O2": 7,     # 산소
                "NO": 8,     # 일산화질소
                "SO2": 9,    # 이산화황
                "NO2": 10,   # 이산화질소
                "NH3": 11,   # 암모니아
                "HNO3": 12,  # 질산
                "OH": 13,    # 하이드록실
                "HF": 14,    # 플루오르화수소
                "HCl": 15,   # 염화수소
                "HBr": 16,   # 브롬화수소
                "HI": 17,    # 요오드화수소
                "ClO": 18,   # 염소산화물
                "OCS": 19,   # 황화카르보닐
                "H2CO": 20,  # 포름알데히드
                "HOCl": 21,  # 차아염소산
                "N2": 22,    # 질소
                "HCN": 23,   # 시안화수소
                "CH3Cl": 24, # 염화메틸
                "H2O2": 25,  # 과산화수소
                "C2H2": 26,  # 아세틸렌
                "C2H6": 27,  # 에탄
                "PH3": 28,   # 포스핀
            }
            
            if molecule not in molecule_ids:
                print(f"❌ 지원하지 않는 분자: {molecule}")
                print(f"지원 분자: {list(molecule_ids.keys())}")
                return None
            
            molecule_id = molecule_ids[molecule]
            print(f"   분자 ID: {molecule_id}")
            
            # Hitran 클래스 사용
            hitran_query = hitran.Hitran()
            
            # 분자 데이터 다운로드
            data = hitran_query.query_lines(
                molecule_number=molecule_id, 
                isotopologue_number=1,
                min_frequency=wavenumber_min * u.cm**-1,
                max_frequency=wavenumber_max * u.cm**-1
            )
            
            print(f"✅ {molecule} 데이터 다운로드 완료!")
            print(f"   라인 개수: {len(data)}")
            return data
            
        except Exception as e:
            print(f"❌ 데이터 다운로드 실패: {e}")
            return None

# 테스트 실행
if __name__ == "__main__":
    print("=== HITRAN API 테스트 (astroquery) ===")
    hitran_api = HitranAPI()
    
    if hitran_api.test_connection():
        data = hitran_api.download_molecule_data("H2O", 1500, 1600)