# 유전자 발현 및 논문 분석 도구

이 프로젝트는 RNA-seq 데이터와 생물학 논문 초록을 분석하여 유전자 발현 패턴과 연구 동향을 시각화하는 도구입니다.

## 주요 기능

### 데이터 처리

- RNA-seq 데이터 로드 및 전처리
- 임상 데이터 통합
- 논문 초록 텍스트 전처리
- 문장 임베딩 생성

### 분석 및 시각화

- 차원 축소 (PCA, t-SNE)
- 유전자 발현 히트맵
- 유전자 간 상관관계 분석
- 임상 변수와 유전자 발현 관계 분석
- 유전자 발현 빈도 분석

## 프로젝트 구조

```

├── 데이터 수집/
│ ├── RNA-seq 데이터/
│ ├── TCGA 데이터/
│ └── 논문 데이터/
├── data/
│ ├── processed_abstracts.txt
│ ├── processed_rnaseq_data.csv
│ └── processed_tcga_clinical_data.csv
└── 자연어처리.py

```

## 필요 라이브러리

- pandas
- numpy
- matplotlib
- seaborn
- nltk
- sentence_transformers
- scikit-learn

## 설치 방법

```bash
pip install pandas numpy matplotlib seaborn nltk sentence-transformers scikit-learn
```

## 사용 방법

1. 필요한 데이터 파일 준비:

   - `data/processed_abstracts.txt`: 전처리된 논문 초록
   - `data/processed_rnaseq_data.csv`: RNA-seq 발현 데이터
   - `data/processed_tcga_clinical_data.csv`: 임상 데이터

2. 스크립트 실행:

```python
python 자연어처리.py
```

## 주요 함수 설명

### 데이터 로드 및 전처리

- `load_and_preprocess_abstracts()`: 논문 초록 로드 및 전처리
- `load_rnaseq_data()`: RNA-seq 데이터 로드
- `load_clinical_data()`: 임상 데이터 로드

### 분석 함수

- `generate_embeddings()`: 텍스트 임베딩 생성
- `compute_similarity_matrix()`: 유사도 매트릭스 계산
- `apply_pca()`: PCA 차원 축소
- `apply_tsne()`: t-SNE 차원 축소

### 시각화 함수

- `visualize_dimension_reduction()`: 차원 축소 결과 시각화
- `visualize_heatmap()`: 유전자 발현 히트맵
- `visualize_correlation_heatmap()`: 상관관계 히트맵
- `visualize_relationship()`: 임상 변수와 유전자 발현 관계
- `visualize_boxplot()`: 그룹별 유전자 발현 비교
- `visualize_expression_frequency()`: 유전자 발현 빈도

## 참고사항

- 한글 폰트 설정이 필요합니다 (NanumGothic 또는 Malgun Gothic)
- NLTK 리소스는 자동으로 다운로드됩니다
- 대용량 데이터 처리 시 메모리 사용량에 주의하세요

## 라이선스

MIT License
