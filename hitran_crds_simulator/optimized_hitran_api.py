"""
최적화된 HITRAN API (캐싱 + 병렬 처리)
"""

from astroquery import hitran
import os
import pandas as pd
import astropy.units as u
import concurrent.futures
import pickle
import hashlib
from datetime import datetime
import time

class HitranCache:
    def __init__(self, cache_dir="cache/hitran_cache"):
        self.cache_dir = cache_dir
        
        # 캐시 폴더 생성
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 캐시 메타데이터 파일
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.load_metadata()
    
    def load_metadata(self):
        """캐시 메타데이터 로드"""
        if os.path.exists(self.metadata_file):
            try:
                self.metadata = pd.read_json(self.metadata_file)
            except:
                self.metadata = pd.DataFrame(columns=['cache_key', 'file_path', 'created_time', 'access_count', 'file_size'])
        else:
            self.metadata = pd.DataFrame(columns=['cache_key', 'file_path', 'created_time', 'access_count', 'file_size'])
    
    def save_metadata(self):
        """캐시 메타데이터 저장"""
        self.metadata.to_json(self.metadata_file, orient='records', date_format='iso')
    
    def generate_cache_key(self, molecule, wavelength_min, wavelength_max):
        """캐시 키 생성"""
        key_string = f"{molecule}_{wavelength_min}_{wavelength_max}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cache_path(self, cache_key):
        """캐시 파일 경로 생성"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def is_cached(self, molecule, wavelength_min, wavelength_max):
        """캐시 존재 여부 확인"""
        cache_key = self.generate_cache_key(molecule, wavelength_min, wavelength_max)
        cache_path = self.get_cache_path(cache_key)
        return os.path.exists(cache_path)
    
    def save_to_cache(self, molecule, wavelength_min, wavelength_max, data):
        """데이터를 캐시에 저장"""
        cache_key = self.generate_cache_key(molecule, wavelength_min, wavelength_max)
        cache_path = self.get_cache_path(cache_key)
        
        try:
            # 데이터 저장
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # 메타데이터 업데이트
            file_size = os.path.getsize(cache_path)
            new_entry = pd.DataFrame({
                'cache_key': [cache_key],
                'file_path': [cache_path],
                'created_time': [datetime.now().isoformat()],
                'access_count': [1],
                'file_size': [file_size],
                'molecule': [molecule],
                'wavelength_min': [wavelength_min],
                'wavelength_max': [wavelength_max]
            })
            
            # 기존 항목 제거 후 추가
            self.metadata = self.metadata[self.metadata['cache_key'] != cache_key]
            self.metadata = pd.concat([self.metadata, new_entry], ignore_index=True)
            self.save_metadata()
            
            return True
        except Exception as e:
            print(f"캐시 저장 실패: {e}")
            return False
    
    def load_from_cache(self, molecule, wavelength_min, wavelength_max):
        """캐시에서 데이터 로드"""
        cache_key = self.generate_cache_key(molecule, wavelength_min, wavelength_max)
        cache_path = self.get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # 접근 횟수 증가
            mask = self.metadata['cache_key'] == cache_key
            if mask.any():
                self.metadata.loc[mask, 'access_count'] += 1
                self.save_metadata()
            
            return data
        except Exception as e:
            print(f"캐시 로드 실패: {e}")
            return None
    
    def get_cache_stats(self):
        """캐시 통계 정보"""
        if len(self.metadata) == 0:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'most_accessed': None,
                'oldest_file': None,
                'cache_hits': 0
            }
        
        total_size = self.metadata['file_size'].sum()
        most_accessed = self.metadata.loc[self.metadata['access_count'].idxmax()]
        
        return {
            'total_files': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'most_accessed': f"{most_accessed['molecule']} ({most_accessed['access_count']}회)",
            'cache_hits': self.metadata['access_count'].sum()
        }

class OptimizedHitranAPI:
    def __init__(self):
        """최적화된 HITRAN API 초기화"""
        # 데이터 폴더 생성
        os.makedirs("cache/", exist_ok=True)
        os.makedirs("data/", exist_ok=True)
        
        # 캐시 관리자 초기화
        self.cache = HitranCache()
        
        # HITRAN 분자 ID 매핑
        self.molecule_ids = {
            "H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6,
            "O2": 7, "NO": 8, "SO2": 9, "NO2": 10, "NH3": 11, "HNO3": 12
        }
    
    def download_molecule_data(self, molecule, wavelength_min, wavelength_max, use_cache=True):
        """분자 데이터 다운로드 (캐싱 지원)"""
        # 캐시 확인
        if use_cache and self.cache.is_cached(molecule, wavelength_min, wavelength_max):
            print(f"🚀 {molecule} 캐시에서 로드 중...")
            data = self.cache.load_from_cache(molecule, wavelength_min, wavelength_max)
            if data is not None:
                print(f"✅ {molecule} 캐시 로드 완료! (라인 수: {len(data)})")
                return data
        
        # 캐시에 없으면 다운로드
        try:
            wavenumber_min = 1e7 / wavelength_max
            wavenumber_max = 1e7 / wavelength_min
            
            print(f"📥 {molecule} 데이터 다운로드 중...")
            
            if molecule not in self.molecule_ids:
                print(f"❌ 지원하지 않는 분자: {molecule}")
                return None
            
            molecule_id = self.molecule_ids[molecule]
            hitran_query = hitran.Hitran()
            
            data = hitran_query.query_lines(
                molecule_number=molecule_id, 
                isotopologue_number=1,
                min_frequency=wavenumber_min * u.cm**-1,
                max_frequency=wavenumber_max * u.cm**-1
            )
            
            print(f"✅ {molecule} 데이터 다운로드 완료! (라인 수: {len(data)})")
            
            # 캐시에 저장
            if use_cache:
                if self.cache.save_to_cache(molecule, wavelength_min, wavelength_max, data):
                    print(f"💾 {molecule} 데이터 캐시에 저장됨")
            
            return data
            
        except Exception as e:
            print(f"❌ 데이터 다운로드 실패: {e}")
            return None
    
    def download_multiple_molecules(self, molecules_params, max_workers=4):
        """
        여러 분자 병렬 다운로드
        
        Args:
            molecules_params: [(molecule, wl_min, wl_max), ...] 리스트
            max_workers: 동시 실행할 최대 스레드 수
        """
        results = {}
        
        print(f"🔄 {len(molecules_params)}개 분자 병렬 다운로드 시작 (최대 {max_workers} 스레드)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 다운로드 작업 제출
            future_to_molecule = {
                executor.submit(self.download_molecule_data, mol, wl_min, wl_max): mol
                for mol, wl_min, wl_max in molecules_params
            }
            
            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_molecule):
                molecule = future_to_molecule[future]
                try:
                    data = future.result()
                    results[molecule] = data
                except Exception as e:
                    print(f"❌ {molecule} 다운로드 실패: {e}")
                    results[molecule] = None
        
        print(f"✅ 병렬 다운로드 완료! 성공: {sum(1 for v in results.values() if v is not None)}/{len(molecules_params)}")
        return results
    
    def get_cache_info(self):
        """캐시 정보 반환"""
        return self.cache.get_cache_stats()

# 테스트 실행
if __name__ == "__main__":
    print("=== 최적화된 HITRAN API 테스트 (병렬 처리 포함) ===")
    
    api = OptimizedHitranAPI()
    
    # === 기본 캐시 테스트 ===
    print("\n1️⃣ 캐시 테스트")
    cache_stats = api.get_cache_info()
    print("초기 캐시 통계:", cache_stats)
    
    data1 = api.download_molecule_data("H2O", 1500, 1520)
    
    # === 병렬 처리 테스트 ===
    print("\n2️⃣ 병렬 처리 테스트")
    molecules = [
        ("H2O", 1500, 1520),   # 물 (캐시됨)
        ("CH4", 1640, 1680),   # 메탄 (새로 다운로드)
        ("CO2", 2000, 2020),   # 이산화탄소 (새로 다운로드)
    ]
    
    start_time = time.time()
    parallel_results = api.download_multiple_molecules(molecules, max_workers=3)
    parallel_time = time.time() - start_time
    
    print(f"\n병렬 다운로드 시간: {parallel_time:.2f}초")
    print("다운로드된 분자:", [mol for mol, data in parallel_results.items() if data is not None])
    
    # === 최종 캐시 통계 ===
    print("\n3️⃣ 최종 캐시 통계")
    final_stats = api.get_cache_info()
    print("최종 캐시 통계:", final_stats)