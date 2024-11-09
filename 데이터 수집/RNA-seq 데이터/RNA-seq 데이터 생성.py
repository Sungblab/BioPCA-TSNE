import numpy as np
import pandas as pd
from scipy.stats import nbinom, multivariate_normal

# 실제 유전자 목록 (암 관련 주요 유전자)
genes = [
    'BRCA1', 'BRCA2', 'TP53', 'EGFR', 'MYC',
    'PTEN', 'RB1', 'KRAS', 'APC', 'PIK3CA',
    'ATM', 'CDH1', 'VHL', 'MLH1', 'MAPK1'
]

# TCGA 샘플 ID 목록
samples = [
    'TCGA-CJ-5678', 'TCGA-AA-3518', 'TCGA-BP-5190',
    'TCGA-D5-6924', 'TCGA-B2-3924', 'TCGA-AA-A00K',
    'TCGA-BP-5200', 'TCGA-B0-4814', 'TCGA-AA-A02E',
    'TCGA-AA-3710', 'TCGA-AA-3819', 'TCGA-CA-6715'
]

# 유전자 간 상관 행렬 설정 (상관관계 예시)
correlation_matrix = np.array([
    [1.0, 0.8, 0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.8, 1.0, 0.4, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.5, 0.4, 1.0, 0.6, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.3, 0.3, 0.6, 1.0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.2, 0.2, 0.3, 0.7, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1.0, 0.5, 0.4, 0.2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.5, 1.0, 0.6, 0.3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.4, 0.6, 1.0, 0.5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.2, 0.3, 0.5, 1.0, 0.3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1.0, 0.7, 0.4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 1.0, 0.6, 0.5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.6, 1.0, 0.3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.3, 1.0, 0.6, 0.4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 1.0, 0.5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.5, 1.0]
])

# 다변량 정규분포를 통해 발현 데이터 생성
mean_expression = np.zeros(len(genes))
expression_data = multivariate_normal.rvs(mean=mean_expression, cov=correlation_matrix, size=len(samples))
expression_data = np.exp(expression_data)  # 양수 값으로 변환

# 발현 값을 특정 범위로 조정 (0~20 사이)
expression_data = np.interp(expression_data, (expression_data.min(), expression_data.max()), (0, 20))

# 데이터프레임 생성 및 전치하여 유전자 x 샘플 형식으로 배치
rnaseq_df = pd.DataFrame(expression_data, index=samples, columns=genes).T
rnaseq_df.index.name = 'Gene'

# CSV 파일로 저장
output_path = 'data/tcga_gene_expression_data.csv'
rnaseq_df.to_csv(output_path)

print(f"RNA-seq 발현 데이터가 {output_path}에 저장되었습니다.")
