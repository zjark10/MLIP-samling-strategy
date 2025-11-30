import os
from mp_api.client import MPRester
from tqdm import tqdm

# Materials Project API Key
API_KEY = "0jc7sZJavHlTPaIGuRuVsjo7deVjmkwe"

# 허용된 원소 리스트
ALLOWED_ELEMENTS = set("""
Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu 
Zn Ga Ge As Se Br Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I 
Ba Lu Hf Ta W Ir Pt Tl Pb La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
""".split())

# 출력 디렉토리
OUTPUT_DIR = "./filtered_structures"

def ensure_directory(directory: str) -> None:
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ 디렉토리 생성: {directory}")

def query_na_o_structures():
    """Materials Project에서 Na와 O를 포함하는 모든 구조 쿼리"""
    print("🔍 Materials Project에서 Na-O 구조 검색 중...")
    
    with MPRester(API_KEY) as mpr:
        results = mpr.materials.summary.search(
            elements=["Na", "O"],
            fields=["material_id", "formula_pretty", "elements"]
        )
    
    print(f"📊 총 {len(results)}개의 Na-O 구조 검색 완료")
    
    # Element 객체를 문자열로 변환
    structures = []
    for result in results:
        structures.append({
            "material_id": result.material_id,
            "formula_pretty": result.formula_pretty,
            "elements": [str(elem) for elem in result.elements]
        })
    
    return structures

def filter_by_allowed_elements(structures):
    """허용된 원소만 포함하는 구조 필터링"""
    print("\n🔬 허용된 원소 기준으로 필터링 중...")
    
    allowed_structures = []
    excluded_structures = []
    
    for struct in tqdm(structures, desc="원소 필터링"):
        elements = set(struct["elements"])
        
        # 모든 원소가 허용 목록에 있는지 확인
        if elements.issubset(ALLOWED_ELEMENTS):
            allowed_structures.append(struct)
        else:
            # 허용되지 않은 원소 찾기
            disallowed = elements - ALLOWED_ELEMENTS
            excluded_structures.append({
                **struct,
                "disallowed_elements": sorted(disallowed)
            })
    
    print(f"✅ 허용: {len(allowed_structures)}개")
    print(f"❌ 제외: {len(excluded_structures)}개")
    
    return allowed_structures, excluded_structures

def save_results(allowed_structures, excluded_structures):
    """결과를 텍스트 파일로 저장"""
    ensure_directory(OUTPUT_DIR)
    
    # 1. 허용된 구조 저장
    allowed_file = os.path.join(OUTPUT_DIR, "1_element_Na_O.txt")
    with open(allowed_file, "w", encoding="utf-8") as f:
        f.write("# Na-O 구조 (허용된 원소만 포함)\n")
        f.write("# material_id\tformula\telements\n")
        f.write("-" * 80 + "\n")
        
        for struct in allowed_structures:
            elements_str = ", ".join(sorted(struct["elements"]))
            f.write(f"{struct['material_id']}\t{struct['formula_pretty']}\t{elements_str}\n")
    
    print(f"\n💾 허용된 구조 저장: {allowed_file}")
    
    # 2. 제외된 구조 저장 (참고용)
    excluded_file = os.path.join(OUTPUT_DIR, "1_element_Na_O_excluded.txt")
    with open(excluded_file, "w", encoding="utf-8") as f:
        f.write("# Na-O 구조 (비허용 원소 포함으로 제외)\n")
        f.write("# material_id\tformula\tdisallowed_elements\n")
        f.write("-" * 80 + "\n")
        
        for struct in excluded_structures:
            disallowed_str = ", ".join(struct["disallowed_elements"])
            f.write(f"{struct['material_id']}\t{struct['formula_pretty']}\t{disallowed_str}\n")
    
    print(f"💾 제외된 구조 저장: {excluded_file}")
    
    # 3. 요약 정보 출력
    print("\n" + "=" * 80)
    print("📊 필터링 결과 요약")
    print("=" * 80)
    print(f"전체 구조:     {len(allowed_structures) + len(excluded_structures):>6}개")
    print(f"허용된 구조:   {len(allowed_structures):>6}개")
    print(f"제외된 구조:   {len(excluded_structures):>6}개")
    print("=" * 80)

def main():
    print("🚀 Na-O 구조 원소 필터링 시작\n")
    
    # 1. Materials Project에서 Na-O 구조 쿼리
    structures = query_na_o_structures()
    
    # 2. 허용된 원소만 포함하는 구조 필터링
    allowed, excluded = filter_by_allowed_elements(structures)
    
    # 3. 결과 저장
    save_results(allowed, excluded)
    
    print("\n✨ 작업 완료!")

if __name__ == "__main__":
    main()
