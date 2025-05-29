"""
최적화된 HITRAN API (캐싱 + 병렬 처리 + 메모리 최적화)
"""

from astroquery import hitran
import os
import pandas as pd
import astropy.units as u
import concurrent.futures
import pickle
import gzip
import hashlib
from datetime import datetime
import time
import gc
import psutil
import numpy as np
from typing import List, Dict, Any, Optional

class MemoryMonitor:
    """메모리 사용량 모니터링"""
    
    @staticmethod
    def get_memory_usage():
        """현재 메모리 사용량 반환 (MB)"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 물리 메모리
            'vms_mb': memory_info.vms / 1024 / 1024,  # 가상 메모리
            'percent': process.memory_percent()        # 시스템 메모리 대비 %
        }
    
    @staticmethod
    def get_system_memory():
        """시스템 전체 메모리 정보"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024 / 1024 / 1024,
            'available_gb': memory.available / 1024 / 1024 / 1024,
            'used_percent': memory.percent
        }
    
    @staticmethod
    def print_memory_status(label=""):
        """메모리 상태 출력"""
        process_mem = MemoryMonitor.get_memory_usage()
        system_mem = MemoryMonitor.get_system_memory()
        
        print(f"🧠 메모리 상태 {label}:")
        print(f"   프로세스: {process_mem['rss_mb']:.1f}MB ({process_mem['percent']:.1f}%)")
        print(f"   시스템: {system_mem['used_percent']:.1f}% 사용중 ({system_mem['available_gb']:.1f}GB 사용가능)")

class OptimizedHitranCache:
    """메모리 최적화된 캐시 시스템"""
    
    def __init__(self, cache_dir="cache/hitran_cache", compression_level=6):
        self.cache_dir = cache_dir
        self.compression_level = compression_level
        
        # 캐시 폴더 생성
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 메타데이터 파일
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.load_metadata()
    
    def load_metadata(self):
        """캐시 메타데이터 로드"""
        if os.path.exists(self.metadata_file):
            try:
                self.metadata = pd.read_json(self.metadata_file)
            except:
                self.metadata = pd.DataFrame(columns=['cache_key', 'file_path', 'created_time', 'access_count', 'file_size', 'compressed_size'])
        else:
            self.metadata = pd.DataFrame(columns=['cache_key', 'file_path', 'created_time', 'access_count', 'file_size', 'compressed_size'])
    
    def save_metadata(self):
        """메타데이터 저장"""
        self.metadata.to_json(self.metadata_file, orient='records', date_format='iso')
    
    def generate_cache_key(self, molecule, wavelength_min, wavelength_max):
        """캐시 키 생성"""
        key_string = f"{molecule}_{wavelength_min}_{wavelength_max}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cache_path(self, cache_key):
        """압축된 캐시 파일 경로"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl.gz")
    
    def is_cached(self, molecule, wavelength_min, wavelength_max):
        """캐시 존재 확인"""
        cache_key = self.generate_cache_key(molecule, wavelength_min, wavelength_max)
        cache_path = self.get_cache_path(cache_key)
        return os.path.exists(cache_path)
    
    def save_to_cache(self, molecule, wavelength_min, wavelength_max, data):
        """압축하여 캐시에 저장"""
        cache_key = self.generate_cache_key(molecule, wavelength_min, wavelength_max)
        cache_path = self.get_cache_path(cache_key)
        
        try:
            # 메모리 상태 확인
            MemoryMonitor.print_memory_status("캐시 저장 전")
            
            # 데이터 압축 저장
            with gzip.open(cache_path, 'wb', compresslevel=self.compression_level) as f:
                pickle.dump(data, f)
            
            # 파일 크기 정보
            compressed_size = os.path.getsize(cache_path)
            
            # 원본 크기 추정 (압축 해제 없이)
            original_size = len(pickle.dumps(data))
            compression_ratio = compressed_size / original_size if original_size > 0 else 0
            
            # 메타데이터 업데이트
            new_entry = pd.DataFrame({
                'cache_key': [cache_key],
                'file_path': [cache_path],
                'created_time': [datetime.now().isoformat()],
                'access_count': [1],
                'file_size': [original_size],
                'compressed_size': [compressed_size],
                'molecule': [molecule],
                'wavelength_min': [wavelength_min],
                'wavelength_max': [wavelength_max],
                'compression_ratio': [compression_ratio]
            })
            
            # 기존 항목 제거 후 추가
            self.metadata = self.metadata[self.metadata['cache_key'] != cache_key]
            self.metadata = pd.concat([self.metadata, new_entry], ignore_index=True)
            self.save_metadata()
            
            print(f"💾 압축 저장: {original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB ({compression_ratio*100:.1f}%)")
            
            # 가비지 컬렉션
            del data
            gc.collect()
            
            MemoryMonitor.print_memory_status("캐시 저장 후")
            
            return True
            
        except Exception as e:
            print(f"❌ 캐시 저장 실패: {e}")
            return False
    
    def load_from_cache(self, molecule, wavelength_min, wavelength_max):
        """압축된 캐시에서 로드"""
        cache_key = self.generate_cache_key(molecule, wavelength_min, wavelength_max)
        cache_path = self.get_cache_path(cache_key)
        
        try:
            MemoryMonitor.print_memory_status("캐시 로드 전")
            
            # 압축 해제하여 로드
            with gzip.open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # 접근 횟수 증가
            mask = self.metadata['cache_key'] == cache_key
            if mask.any():
                self.metadata.loc[mask, 'access_count'] += 1
                self.save_metadata()
            
            MemoryMonitor.print_memory_status("캐시 로드 후")
            
            return data
            
        except Exception as e:
            print(f"❌ 캐시 로드 실패: {e}")
            return None
    
    def get_cache_stats(self):
        """캐시 통계 (압축 정보 포함)"""
        if len(self.metadata) == 0:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'compressed_size_mb': 0,
                'compression_ratio': 0,
                'most_accessed': None,
                'cache_hits': 0
            }
        
        total_size = self.metadata['file_size'].sum() if 'file_size' in self.metadata.columns else 0
        compressed_size = self.metadata['compressed_size'].sum() if 'compressed_size' in self.metadata.columns else 0
        
        most_accessed = self.metadata.loc[self.metadata['access_count'].idxmax()]
        avg_compression = compressed_size / total_size if total_size > 0 else 0
        
        return {
            'total_files': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': avg_compression,
            'space_saved_mb': (total_size - compressed_size) / (1024 * 1024),
            'most_accessed': f"{most_accessed['molecule']} ({most_accessed['access_count']}회)",
            'cache_hits': self.metadata['access_count'].sum()
        }
    
    def cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        MemoryMonitor.print_memory_status("메모리 정리 후")

class MemoryOptimizedHitranAPI:
    """메모리 최적화된 HITRAN API"""
    
    def __init__(self, max_memory_mb=1000):
        """초기화"""
        self.max_memory_mb = max_memory_mb
        
        # 폴더 생성
        os.makedirs("cache/", exist_ok=True)
        os.makedirs("data/", exist_ok=True)
        
        # 최적화된 캐시 시스템
        self.cache = OptimizedHitranCache()
        
        # 분자 ID 매핑
        self.molecule_ids = {
            "H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6,
            "O2": 7, "NO": 8, "SO2": 9, "NO2": 10, "NH3": 11, "HNO3": 12
        }
        
        MemoryMonitor.print_memory_status("API 초기화")
    
    def check_memory_limit(self):
        """메모리 한계 확인"""
        memory_usage = MemoryMonitor.get_memory_usage()
        if memory_usage['rss_mb'] > self.max_memory_mb:
            print(f"⚠️ 메모리 한계 초과: {memory_usage['rss_mb']:.1f}MB > {self.max_memory_mb}MB")
            print("🧹 가비지 컬렉션 실행...")
            gc.collect()
            return False
        return True
    
    def download_molecule_data_chunked(self, molecule, wavelength_min, wavelength_max, chunk_size=100, use_cache=True):
        """청크 단위로 분자 데이터 다운로드"""
        
        # 캐시 확인
        if use_cache and self.cache.is_cached(molecule, wavelength_min, wavelength_max):
            print(f"🚀 {molecule} 캐시에서 로드 중...")
            data = self.cache.load_from_cache(molecule, wavelength_min, wavelength_max)
            if data is not None:
                print(f"✅ {molecule} 캐시 로드 완료! (라인 수: {len(data)})")
                return data
        
        # 메모리 확인
        if not self.check_memory_limit():
            print("❌ 메모리 부족으로 다운로드 중단")
            return None
        
        try:
            wavenumber_min = 1e7 / wavelength_max
            wavenumber_max = 1e7 / wavelength_min
            
            print(f"📥 {molecule} 데이터 다운로드 중 (청크 크기: {chunk_size})")
            MemoryMonitor.print_memory_status("다운로드 시작")
            
            if molecule not in self.molecule_ids:
                print(f"❌ 지원하지 않는 분자: {molecule}")
                return None
            
            molecule_id = self.molecule_ids[molecule]
            hitran_query = hitran.Hitran()
            
            # 데이터 다운로드
            data = hitran_query.query_lines(
                molecule_number=molecule_id, 
                isotopologue_number=1,
                min_frequency=wavenumber_min * u.cm**-1,
                max_frequency=wavenumber_max * u.cm**-1
            )
            
            print(f"✅ {molecule} 데이터 다운로드 완료! (라인 수: {len(data)})")
            MemoryMonitor.print_memory_status("다운로드 완료")
            
            # 캐시에 저장
            if use_cache:
                self.cache.save_to_cache(molecule, wavelength_min, wavelength_max, data)
            
            return data
            
        except Exception as e:
            print(f"❌ 데이터 다운로드 실패: {e}")
            return None
        finally:
            # 메모리 정리
            gc.collect()
    
    def download_multiple_molecules_optimized(self, molecules_params, max_workers=3):
        """메모리 최적화된 병렬 다운로드"""
        results = {}
        
        print(f"🔄 {len(molecules_params)}개 분자 메모리 최적화 병렬 다운로드")
        MemoryMonitor.print_memory_status("병렬 다운로드 시작")
        
        # 메모리 사용량을 고려하여 worker 수 조정
        memory_info = MemoryMonitor.get_system_memory()
        if memory_info['available_gb'] < 2:  # 2GB 미만이면 worker 수 감소
            max_workers = min(max_workers, 2)
            print(f"⚠️ 메모리 부족으로 worker 수 조정: {max_workers}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_molecule = {
                executor.submit(self.download_molecule_data_chunked, mol, wl_min, wl_max): mol
                for mol, wl_min, wl_max in molecules_params
            }
            
            for future in concurrent.futures.as_completed(future_to_molecule):
                molecule = future_to_molecule[future]
                try:
                    data = future.result()
                    results[molecule] = data
                    
                    # 주기적 메모리 정리
                    if not self.check_memory_limit():
                        print(f"🧹 메모리 정리 (처리 완료: {molecule})")
                        
                except Exception as e:
                    print(f"❌ {molecule} 다운로드 실패: {e}")
                    results[molecule] = None
        
        # 최종 메모리 정리
        self.cache.cleanup_memory()
        
        print(f"✅ 메모리 최적화 병렬 다운로드 완료! 성공: {sum(1 for v in results.values() if v is not None)}/{len(molecules_params)}")
        MemoryMonitor.print_memory_status("병렬 다운로드 완료")
        
        return results
    
    def get_optimization_stats(self):
        """최적화 통계 반환"""
        cache_stats = self.cache.get_cache_stats()
        memory_stats = MemoryMonitor.get_memory_usage()
        system_stats = MemoryMonitor.get_system_memory()
        
        return {
            'cache': cache_stats,
            'memory': memory_stats,
            'system': system_stats
        }

# 테스트 실행
if __name__ == "__main__":
    print("=== 메모리 최적화된 HITRAN API 테스트 ===")
    
    # 시스템 정보
    system_info = MemoryMonitor.get_system_memory()
    print(f"💻 시스템 메모리: {system_info['total_gb']:.1f}GB (사용가능: {system_info['available_gb']:.1f}GB)")
    
    api = MemoryOptimizedHitranAPI(max_memory_mb=500)  # 500MB 제한
    
    # === 메모리 최적화 테스트 ===
    print("\n1️⃣ 단일 분자 테스트 (메모리 최적화)")
    data1 = api.download_molecule_data_chunked("H2O", 1500, 1520)
    
    # === 병렬 + 메모리 최적화 테스트 ===
    print("\n2️⃣ 병렬 + 메모리 최적화 테스트")
    molecules = [
        ("H2O", 1500, 1520),   # 캐시됨
        ("CH4", 1640, 1680),   # 새로 다운로드
        ("CO2", 2000, 2020),   # 새로 다운로드
        ("NH3", 1450, 1470),   # 새로 다운로드
    ]
    
    start_time = time.time()
    results = api.download_multiple_molecules_optimized(molecules, max_workers=3)
    total_time = time.time() - start_time
    
    print(f"\n⚡ 총 처리 시간: {total_time:.2f}초")
    print(f"다운로드 성공: {[mol for mol, data in results.items() if data is not None]}")
    
    # === 최적화 통계 ===
    print("\n3️⃣ 최적화 통계")
    stats = api.get_optimization_stats()
    
    print("📊 캐시 통계:")
    cache = stats['cache']
    print(f"   파일 수: {cache['total_files']}개")
    print(f"   원본 크기: {cache['total_size_mb']:.2f}MB")
    print(f"   압축 크기: {cache['compressed_size_mb']:.2f}MB")
    print(f"   압축률: {cache['compression_ratio']*100:.1f}%")
    print(f"   절약 공간: {cache['space_saved_mb']:.2f}MB")
    
    print("🧠 최종 메모리 상태:")
    memory = stats['memory']
    print(f"   프로세스 메모리: {memory['rss_mb']:.1f}MB")
    print(f"   시스템 사용률: {stats['system']['used_percent']:.1f}%")