import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Optional
import numpy as np

def preprocess_tcga_clinical_xml(xml_file: str) -> List[Dict]:
    """
    TCGA 임상 XML 데이터를 전처리하는 함수
    
    Parameters:
        xml_file (str): XML 파일 경로
        
    Returns:
        List[Dict]: 전처리된 환자 데이터 리스트
    """
    # 네임스페이스 정의
    namespaces = {
        'ns0': 'http://tcga.nci/bcr/xml/administration/2.7',
        'ns2': 'http://tcga.nci/bcr/xml/clinical/kirc/2.7',
        'ns3': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
        'ns4': 'http://tcga.nci/bcr/xml/shared/2.7',
        'ns5': 'http://tcga.nci/bcr/xml/clinical/shared/stage/2.7',
        'ns9': 'http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7',
        'ns12': 'http://tcga.nci/bcr/xml/clinical/coad/2.7'
    }
    
    # XML 파싱
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 모든 clinical_data 요소 찾기
    clinical_data_list = []
    
    for clinical_data in root.findall('clinical_data'):
        patient_data = {}
        
        # 관리 정보 추출
        admin = clinical_data.find('.//ns0:admin', namespaces)
        if admin is not None:
            patient_data['project_code'] = get_element_text(admin, './/ns0:project_code', namespaces)
            patient_data['disease_code'] = get_element_text(admin, './/ns0:disease_code', namespaces)
            patient_data['batch_number'] = get_element_text(admin, './/ns0:batch_number', namespaces)
        
        # 환자 정보 추출 - KIRC와 COAD 모두 고려
        patient = clinical_data.find('.//ns2:patient', namespaces) or clinical_data.find('.//ns12:patient', namespaces)
        if patient is not None:
            # 기본 정보
            patient_data.update({
                'bcr_patient_barcode': get_element_text(patient, './/ns4:bcr_patient_barcode', namespaces),
                'gender': get_element_text(patient, './/ns4:gender', namespaces),
                'vital_status': get_element_text(patient, './/ns3:vital_status', namespaces),
                'days_to_birth': safe_int(get_element_text(patient, './/ns3:days_to_birth', namespaces)),
                'days_to_death': safe_int(get_element_text(patient, './/ns3:days_to_death', namespaces)),
                'days_to_last_followup': safe_int(get_element_text(patient, './/ns3:days_to_last_followup', namespaces)),
                
                # 진단 정보
                'tumor_tissue_site': get_element_text(patient, './/ns3:tumor_tissue_site', namespaces),
                'histological_type': get_element_text(patient, './/ns4:histological_type', namespaces),
                'tumor_grade': get_element_text(patient, './/ns4:neoplasm_histologic_grade', namespaces),
                'age_at_diagnosis': safe_int(get_element_text(patient, './/ns3:age_at_initial_pathologic_diagnosis', namespaces)),
                'year_of_diagnosis': get_element_text(patient, './/ns3:year_of_initial_pathologic_diagnosis', namespaces),
                
                # 인종/민족 정보
                'race': get_race(patient, namespaces),
                'ethnicity': get_element_text(patient, './/ns3:ethnicity', namespaces)
            })
            
            # TNM 병기 정보
            stage_event = patient.find('.//ns5:stage_event', namespaces)
            if stage_event is not None:
                patient_data.update({
                    'pathologic_stage': get_element_text(stage_event, './/ns5:pathologic_stage', namespaces),
                    'pathologic_T': get_element_text(stage_event, './/ns5:pathologic_T', namespaces),
                    'pathologic_N': get_element_text(stage_event, './/ns5:pathologic_N', namespaces),
                    'pathologic_M': get_element_text(stage_event, './/ns5:pathologic_M', namespaces)
                })
            
            # 치료 정보
            patient_data.update({
                'radiation_therapy': get_element_text(patient, './/ns3:radiation_therapy', namespaces),
                'neoadjuvant_therapy': get_element_text(patient, './/ns4:history_of_neoadjuvant_treatment', namespaces)
            })
            
            # 약물 치료 정보 추출
            drugs = extract_drug_info(patient, namespaces)
            if drugs:
                # 첫 번째 약물 정보만 기본 필드에 포함
                first_drug = drugs[0]
                for key, value in first_drug.items():
                    patient_data[f'drug_{key}'] = value
                
                # 추가 약물이 있는 경우 별도 필드로 저장
                if len(drugs) > 1:
                    patient_data['additional_drugs'] = drugs[1:]
            
        clinical_data_list.append(patient_data)
    
    return clinical_data_list

def get_element_text(element: ET.Element, xpath: str, namespaces: Dict) -> Optional[str]:
    """XML 요소에서 텍스트를 안전하게 추출"""
    elem = element.find(xpath, namespaces)
    return elem.text if elem is not None else None

def get_race(patient: ET.Element, namespaces: Dict) -> Optional[str]:
    """인종 정보 추출"""
    race_list = patient.find('.//ns3:race_list', namespaces)
    if race_list is not None:
        race = race_list.find('.//ns3:race', namespaces)
        return race.text if race is not None else None
    return None

def extract_drug_info(patient: ET.Element, namespaces: Dict) -> List[Dict]:
    """약물 치료 정보 추출"""
    drugs = patient.findall('.//ns9:drug', namespaces)
    drug_info = []
    
    for drug in drugs:
        drug_data = {
            'name': get_element_text(drug, './/ns9:drug_name', namespaces),
            'prescribed_dose': get_element_text(drug, './/ns9:prescribed_dose', namespaces),
            'prescribed_dose_units': get_element_text(drug, './/ns9:prescribed_dose_units', namespaces),
            'number_cycles': safe_int(get_element_text(drug, './/ns9:number_cycles', namespaces)),
            'therapy_start_days': safe_int(get_element_text(drug, './/ns9:days_to_drug_therapy_start', namespaces)),
            'therapy_end_days': safe_int(get_element_text(drug, './/ns9:days_to_drug_therapy_end', namespaces)),
            'therapy_ongoing': get_element_text(drug, './/ns9:therapy_ongoing', namespaces)
        }
        drug_info.append(drug_data)
    
    return drug_info

def safe_int(value: str) -> Optional[int]:
    """문자열을 안전하게 정수로 변환"""
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None

def save_to_csv(data: List[Dict], output_file: str) -> pd.DataFrame:
    """데이터를 CSV 파일로 저장"""
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    # 추가 약물 정보 컬럼 제외하고 저장 (필요한 경우 별도 처리 가능)
    if 'additional_drugs' in df.columns:
        df = df.drop('additional_drugs', axis=1)
    
    # CSV 파일로 저장
    df.to_csv(output_file, index=False)
    print(f"데이터가 {output_file}에 저장되었습니다.")
    
    return df

if __name__ == "__main__":
    # XML 파일 처리
    xml_file = "merged_clinical_data.xml"
    output_file = "processed_tcga_clinical_data.csv"
    
    try:
        # 데이터 전처리
        print("XML 파일 처리 중...")
        clinical_data = preprocess_tcga_clinical_xml(xml_file)
        
        # CSV로 저장
        print("\nCSV 파일 생성 중...")
        df = save_to_csv(clinical_data, output_file)
        
        # 기본 통계 출력
        print("\n기본 통계:")
        print(f"총 환자 수: {len(df)}")
        print("\n각 질병 코드별 환자 수:")
        print(df['disease_code'].value_counts())
        print("\n생존 상태:")
        print(df['vital_status'].value_counts())
        
        # 데이터 품질 체크
        print("\n결측치 비율:")
        missing_rates = (df.isnull().sum() / len(df) * 100).round(2)
        print(missing_rates[missing_rates > 0])
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise e  # 디버깅을 위해 전체 에러 트레이스백 출력