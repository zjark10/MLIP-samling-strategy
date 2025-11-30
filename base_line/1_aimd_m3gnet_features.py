#!/usr/bin/env python3
"""
M3GNet 기반 특성 추출: uniform_sampled_500structures_200fs.xyz 파일 처리
ExtXYZ 형식의 구조들에 대해 M3GNet을 사용하여 구조 특성을 추출하고 HDF5로 저장

Usage:
python extract_m3gnet_features.py --input uniform_sampled_500structures_200fs.xyz --output features.h5 --batch_size 32 --n_jobs 4
"""

import os
import sys
import numpy as np
import h5py
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import warnings

# TensorFlow 경고 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from ase.io import read
from maml.describers import M3GNetStructure

def read_extxyz_structures(filename):
    """ExtXYZ 파일에서 모든 구조 읽기"""
    try:
        structures = read(filename, index=':')
        print(f"총 {len(structures)}개의 구조를 로드했습니다.")
        return structures
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return []

def process_structure_batch(batch_data):
    """구조 배치에 대한 M3GNet 특성 추출 (멀티프로세싱용)"""
    batch_structures, batch_indices, batch_id = batch_data
    
    try:
        # 각 프로세스에서 새로운 M3GNet 인코더 생성
        structure_encoder = M3GNetStructure()
        
        # 배치 단위로 특성 추출
        structure_features = structure_encoder.transform(batch_structures)
        
        # 결과 정리
        results = []
        for idx, (structure_idx, structure_feat) in enumerate(zip(batch_indices, structure_features)):
            results.append({
                'structure_index': structure_idx,
                'features': structure_feat,
                'num_atoms': len(batch_structures[idx])
            })
        
        print(f"배치 {batch_id}: {len(results)}개 구조 처리 완료")
        return results
        
    except Exception as e:
        print(f"배치 {batch_id} 처리 중 오류: {e}")
        return []

def extract_features_parallel(structures, batch_size=32, n_jobs=4):
    """병렬로 M3GNet 특성 추출"""
    total_structures = len(structures)
    print(f"총 {total_structures}개 구조를 {batch_size} 크기의 배치로 나누어 {n_jobs}개 프로세스로 처리합니다.")
    
    # 배치 준비
    batches = []
    for i in range(0, total_structures, batch_size):
        end_idx = min(i + batch_size, total_structures)
        batch_structures = structures[i:end_idx]
        batch_indices = list(range(i, end_idx))
        batch_id = len(batches) + 1
        batches.append((batch_structures, batch_indices, batch_id))
    
    print(f"총 {len(batches)}개의 배치 생성")
    
    # 병렬 처리
    all_features = []
    
    if n_jobs == 1:
        # 단일 프로세스 처리 (디버깅용)
        for batch_data in tqdm(batches, desc="배치 처리"):
            batch_results = process_structure_batch(batch_data)
            all_features.extend(batch_results)
    else:
        # 멀티프로세싱
        with mp.Pool(processes=n_jobs) as pool:
            batch_results = list(tqdm(
                pool.imap(process_structure_batch, batches),
                total=len(batches),
                desc="배치 처리"
            ))
            
            # 결과 병합
            for batch_result in batch_results:
                all_features.extend(batch_result)
    
    # 구조 인덱스 순으로 정렬
    all_features.sort(key=lambda x: x['structure_index'])
    
    print(f"총 {len(all_features)}개 구조의 특성 추출 완료")
    return all_features

def save_features_to_hdf5(features, output_file):
    """추출된 특성을 HDF5 파일로 저장"""
    if not features:
        print("저장할 특성이 없습니다.")
        return
    
    with h5py.File(output_file, 'w') as f:
        # 메타데이터 그룹 생성
        meta_group = f.create_group('metadata')
        meta_group.attrs['num_structures'] = len(features)
        meta_group.attrs['feature_dim'] = features[0]['features'].shape[0]
        meta_group.attrs['description'] = 'M3GNet structure features from ExtXYZ file'
        
        # 구조 인덱스 저장
        structure_indices = [feat['structure_index'] for feat in features]
        meta_group.create_dataset('structure_indices', data=np.array(structure_indices))
        
        # 원자 수 저장
        num_atoms = [feat['num_atoms'] for feat in features]
        meta_group.create_dataset('num_atoms', data=np.array(num_atoms))
        
        # 모든 특성을 단일 2D 배열로 저장 (압축 적용)
        all_features = np.vstack([feat['features'] for feat in features])
        f.create_dataset('features', data=all_features, compression='gzip', compression_opts=9)
        
        print(f"특성 데이터가 {output_file}에 저장되었습니다.")
        print(f"- 구조 수: {len(features)}")
        print(f"- 특성 차원: {features[0]['features'].shape[0]}")
        print(f"- 데이터 크기: {all_features.shape}")

def validate_structures(structures):
    """구조 데이터 유효성 검사"""
    if not structures:
        print("오류: 구조가 로드되지 않았습니다.")
        return False
    
    print(f"구조 유효성 검사:")
    print(f"- 총 구조 수: {len(structures)}")
    
    # 첫 번째 구조 정보 출력
    first_structure = structures[0]
    print(f"- 첫 번째 구조 원자 수: {len(first_structure)}")
    print(f"- 원소 종류: {set(first_structure.get_chemical_symbols())}")
    print(f"- 셀 정보: {first_structure.cell}")
    print(f"- PBC: {first_structure.pbc}")
    
    # 구조별 원자 수 분포 확인
    atom_counts = [len(structure) for structure in structures]
    print(f"- 원자 수 범위: {min(atom_counts)} ~ {max(atom_counts)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='ExtXYZ 파일에서 M3GNet 구조 특성 추출'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='입력 ExtXYZ 파일 경로')
    parser.add_argument('--output', type=str, default='m3gnet_features.h5',
                       help='출력 HDF5 파일 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기 (기본값: 32)')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='병렬 처리 프로세스 수 (기본값: 4)')
    parser.add_argument('--validate_only', action='store_true',
                       help='구조 유효성 검사만 수행')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"오류: 입력 파일 {args.input}이 존재하지 않습니다.")
        sys.exit(1)
    
    print(f"M3GNet 특성 추출 시작")
    print(f"입력 파일: {args.input}")
    print(f"출력 파일: {args.output}")
    print(f"배치 크기: {args.batch_size}")
    print(f"병렬 프로세스 수: {args.n_jobs}")
    
    # 구조 로드
    print("\n구조 로딩 중...")
    structures = read_extxyz_structures(args.input)
    
    # 구조 유효성 검사
    if not validate_structures(structures):
        sys.exit(1)
    
    if args.validate_only:
        print("유효성 검사 완료. 종료합니다.")
        return
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # M3GNet 특성 추출
    print(f"\nM3GNet 특성 추출 시작...")
    features = extract_features_parallel(
        structures, 
        batch_size=args.batch_size, 
        n_jobs=args.n_jobs
    )
    
    # 결과 저장
    if features:
        print(f"\n결과 저장 중...")
        save_features_to_hdf5(features, args.output)
        print(f"\n처리 완료!")
    else:
        print("오류: 추출된 특성이 없습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()
