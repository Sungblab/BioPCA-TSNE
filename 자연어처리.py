import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
import matplotlib.font_manager as fm

# 경고 메시지 무시 (선택 사항)
warnings.filterwarnings('ignore')

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'  # 또는 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Seaborn 테마 설정
sns.set_theme(font='NanumGothic', style='whitegrid')

def load_and_preprocess_abstracts(path):
    """
    파일에서 초록을 로드하고 전처리하는 함수
    """
    try:  
        with open(path, 'r', encoding='utf-8') as file:
            abstracts = file.readlines()
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    
    # 데이터프레임으로 변환
    abstracts_df = pd.DataFrame({'abstract': [abstract.strip() for abstract in abstracts]})
    
    # 텍스트 전처리 함수 정의
    def preprocess_text(text):
        if pd.isnull(text):
            return ""
        # 소문자 변환
        text = text.lower()
        # 구두점 제거
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 토큰화
        tokens = word_tokenize(text)
        # 불용어 제거
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    # 전처리 적용
    abstracts_df['processed_abstract'] = abstracts_df['abstract'].apply(preprocess_text)
    
    return abstracts_df

def generate_embeddings(df, model_name='all-MiniLM-L6-v2'):
    """
    전처리된 텍스트를 임베딩 벡터로 변환하는 함수
    """
    if df.empty or 'processed_abstract' not in df.columns:
        print("임베딩을 생성할 데이터가 없습니다.")
        return df, np.array([])
    
    # SentenceTransformer 모델 로드
    model = SentenceTransformer(model_name)
    
    # 임베딩 생성 (배치 처리로 메모리 효율성 향상)
    embeddings = model.encode(df['processed_abstract'].tolist(), show_progress_bar=True)
    
    # 임베딩을 데이터프레임에 추가
    df['embedding'] = embeddings.tolist()
    
    return df, embeddings

def load_rnaseq_data(path):
    """
    RNA-seq 데이터를 로드하는 함수.
    첫 번째 행을 유전자 ID로 설정하고, 첫 번째 열을 샘플 ID로 설정합니다.
    """
    try:
        # 첫 번째 행을 헤더로 사용하고, 첫 번째 열을 인덱스로 설정
        rnaseq_df = pd.read_csv(path, index_col=0)
        print("RNA-seq 데이터 로드 성공!")
        print("데이터 형태:", rnaseq_df.shape)
        print("첫 5행 데이터:")
        print(rnaseq_df.head())
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        print(f"CSV 파싱 오류: {e}")
        return pd.DataFrame()

    if rnaseq_df.empty:
        print("RNA-seq 데이터가 비어 있습니다.")
        return rnaseq_df

    # 데이터 전치 (유전자가 행, 샘플이 열이 되도록)
    rnaseq_df = rnaseq_df.transpose()

    # 유전자 발현 데이터를 숫자형으로 변환
    rnaseq_df = rnaseq_df.apply(pd.to_numeric, errors='coerce')

    # 결측치 처리: 모든 값이 NaN인 유전자 제거, 결측치가 있는 샘플 제거
    rnaseq_df = rnaseq_df.dropna(axis=1, how='all')  # 모든 값이 NaN인 유전자 제거
    rnaseq_df = rnaseq_df.dropna(axis=0, how='any')  # 결측치가 있는 샘플 제거

    print("전처리 후 RNA-seq 데이터 형태:", rnaseq_df.shape)
    return rnaseq_df

def load_clinical_data(path):
    """
    임상 데이터를 로드하는 함수
    """
    try:
        clinical_df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    
    # 'bcr_patient_barcode'가 존재하는지 확인
    if 'bcr_patient_barcode' not in clinical_df.columns:
        print("'bcr_patient_barcode' 컬럼이 임상 데이터에 존재하지 않습니다.")
        return pd.DataFrame()
    
    # 결측치 처리: 'bcr_patient_barcode'가 없는 행 제거
    clinical_df = clinical_df.dropna(subset=['bcr_patient_barcode'])
    
    return clinical_df

def merge_rnaseq_clinical(rnaseq_df, clinical_df):
    """
    RNA-seq 데이터와 임상 데이터를 병합하는 함수
    """
    if rnaseq_df.empty or clinical_df.empty:
        print("병합할 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 임상 데이터에서 'bcr_patient_barcode'를 인덱스로 설정
    clinical_df = clinical_df.set_index('bcr_patient_barcode')
    
    # RNA-seq 데이터와 임상 데이터 병합 (내부 조인)
    merged_df = rnaseq_df.merge(clinical_df, left_index=True, right_index=True, how='inner')
    
    return merged_df

def compute_similarity_matrix(embeddings):
    """
    임베딩 벡터 간의 코사인 유사도 매트릭스를 계산하는 함수
    """
    if embeddings.size == 0:
        print("유사도 매트릭스를 계산할 임베딩이 없습니다.")
        return np.array([])
    
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def apply_pca(embeddings, n_components=2):
    """
    PCA를 적용하여 임베딩 벡터의 차원을 축소하는 함수
    """
    if embeddings.size == 0:
        print("PCA를 적용할 임베딩이 없습니다.")
        return np.array([])
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(embeddings)
    return pca_result

def apply_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000):
    """
    t-SNE를 적용하여 임베딩 벡터의 차원을 축소하는 함수
    """
    if embeddings.size == 0:
        print("t-SNE를 적용할 임베딩이 없습니다.")
        return np.array([])
    
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, n_iter=n_iter)
    tsne_result = tsne.fit_transform(embeddings)
    return tsne_result

def visualize_dimension_reduction(df, pca_result, tsne_result, labels=None):
    """
    PCA와 t-SNE 결과를 시각화하는 함수 (한글 버전)
    """
    if pca_result.size > 0:
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]
    if tsne_result.size > 0:
        df['tSNE1'] = tsne_result[:, 0]
        df['tSNE2'] = tsne_result[:, 1]
    
    plt.figure(figsize=(16, 7))
    
    # PCA 시각화
    plt.subplot(1, 2, 1)
    if labels is not None and labels in df.columns:
        sns.scatterplot(x='PCA1', y='PCA2', hue=labels, palette='viridis', data=df, s=50, alpha=0.7)
        plt.legend(title='유전자 이름', loc='best')
    else:
        sns.scatterplot(x='PCA1', y='PCA2', data=df, s=50, alpha=0.7)
    plt.title('초록 임베딩의 PCA 결과')
    plt.xlabel('주성분 1')
    plt.ylabel('주성분 2')
    plt.text(0.05, 0.95, 'PCA: 데이터의 분산을 최대한 보존하는 선형 변환',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    # t-SNE 시각화
    plt.subplot(1, 2, 2)
    if labels is not None and labels in df.columns:
        sns.scatterplot(x='tSNE1', y='tSNE2', hue=labels, palette='viridis', data=df, s=50, alpha=0.7)
        plt.legend(title='유전자 이름', loc='best')
    else:
        sns.scatterplot(x='tSNE1', y='tSNE2', data=df, s=50, alpha=0.7)
    plt.title('초록 임베딩의 t-SNE 결과')
    plt.xlabel('t-SNE 1차원')
    plt.ylabel('t-SNE 2차원')
    plt.text(0.05, 0.95, 't-SNE: 데이터의 국소 구조를 보존하는 비선형 차원 축소',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def visualize_heatmap(rnaseq_df, top_n=50):
    """
    유전자 발현 데이터의 히트맵을 시각화하는 함수 (한글 버전)
    """
    if rnaseq_df.empty:
        print("히트맵을 시각화할 RNA-seq 데이터가 없습니다.")
        return
    
    # 상위 n개의 변동성이 큰 유전자 선택
    variances = rnaseq_df.var().sort_values(ascending=False)
    top_genes = variances.head(top_n).index
    
    # 상위 유전자 데이터 추출
    top_genes_data = rnaseq_df[top_genes]
    
    if top_genes_data.empty:
        print(f"상위 {top_n} 유전자에 대한 데이터가 없습니다.")
        return
    
    # 히트맵 시각화
    plt.figure(figsize=(15, 10))
    sns.heatmap(top_genes_data.T, cmap='viridis', xticklabels=False, yticklabels=True)
    plt.title(f'상위 {top_n} 변동성 유전자 발현 히트맵')
    plt.xlabel('샘플')
    plt.ylabel('유전자')
    plt.text(0.5, 1.05, '히트맵: 유전자 발현 패턴의 시각화',
             fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.show()

def visualize_correlation_heatmap(rnaseq_df, top_n=30):
    """
    유전자 발현 데이터의 상관관계 히트맵을 시각화하는 함수 (한글 버전)
    """
    if rnaseq_df.empty:
        print("상관관계 히트맵을 시각화할 RNA-seq 데이터가 없습니다.")
        return
    
    # 상위 n개의 유전자 선택
    variances = rnaseq_df.var().sort_values(ascending=False)
    top_genes = variances.head(top_n).index
    selected_data = rnaseq_df[top_genes]
    
    # 상관관계 계산
    correlation_matrix = selected_data.corr()
    
    if correlation_matrix.empty:
        print(f"상위 {top_n} 유전자 간의 상관관계 데이터를 계산할 수 없습니다.")
        return
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f")
    plt.title('상위 유전자 간의 상관관계 매트릭스')
    plt.xlabel('유전자')
    plt.ylabel('유전자')
    plt.text(0.5, 1.05, '상관관계 히트맵: 유전자 간 발현 상관관계 시각화',
             fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.show()

def visualize_relationship(merged_df, clinical_var, gene):
    """
    임상 변수와 유전자 발현 간의 관계를 산점도로 시각화하는 함수 (한글 버전 및 추가 설명)
    """
    if gene not in merged_df.columns:
        print(f"'{gene}' 유전자가 데이터에 존재하지 않습니다.")
        return
    if clinical_var not in merged_df.columns:
        print(f"'{clinical_var}' 임상 변수가 데이터에 존재하지 않습니다.")
        return
    
    # 결측치 제거
    plot_df = merged_df.dropna(subset=[clinical_var, gene])
    
    if plot_df.empty:
        print(f"'{clinical_var}'와 '{gene}'에 대한 데이터가 없습니다.")
        return
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=clinical_var, y=gene, data=plot_df, alpha=0.6)
    sns.regplot(x=clinical_var, y=gene, data=plot_df, scatter=False, color='red')
    
    # 상관계수 계산
    corr = plot_df[clinical_var].corr(plot_df[gene])
    
    plt.title(f'{gene} 발현량과 {clinical_var}의 관계\n상관계수: {corr:.2f}')
    plt.xlabel(clinical_var)
    plt.ylabel(f'{gene}의 발현량')
    plt.text(0.05, 0.95, '상관분석: 두 변수 간의 관계를 측정',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.show()

def visualize_boxplot(merged_df, group_var, gene):
    """
    그룹 간 유전자 발현 수준을 박스플롯으로 시각화하는 함수 (한글 버전)
    """
    if gene not in merged_df.columns:
        print(f"'{gene}' 유전자가 데이터에 존재하지 않습니다.")
        return
    if group_var not in merged_df.columns:
        print(f"'{group_var}' 그룹 변수가 데이터에 존재하지 않습니다.")
        return
    
    # 결측치 제거
    plot_df = merged_df.dropna(subset=[group_var, gene])
    
    if plot_df.empty:
        print(f"'{group_var}'와 '{gene}'에 대한 데이터가 없습니다.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_var, y=gene, data=plot_df)
    plt.title(f'{group_var}에 따른 {gene} 발현 수준 비교')
    plt.xlabel(group_var)
    plt.ylabel(f'{gene}의 발현량')
    plt.xticks(rotation=45)
    plt.text(0.05, 0.95, '박스플롯: 데이터 분포의 요약 통계 시각화',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.show()

def visualize_expression_frequency(rnaseq_df, top_n=20):
    """
    유전자 발현의 평균 빈도를 바 플롯으로 시각화하는 함수 (한글 버전)
    """
    if rnaseq_df.empty:
        print("발현 빈도 시각화를 위한 RNA-seq 데이터가 없습니다.")
        return
    
    mean_expression = rnaseq_df.mean().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x=mean_expression.values, y=mean_expression.index, palette='viridis')
    plt.title(f'평균 발현 수준 상위 {top_n} 유전자')
    plt.xlabel('평균 발현량')
    plt.ylabel('유전자')
    plt.text(0.5, 1.05, '바 플롯: 범주형 데이터의 값 비교 시각화',
             fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.show()

def visualize_gene_occurrence_frequency(df, gene_col='gene_name', top_n=20):
    """
    초록에서 특정 유전자의 언급 빈도를 바 플롯으로 시각화하는 함수 (한글 버전)
    """
    if gene_col not in df.columns:
        print(f"'{gene_col}' 컬럼이 데이터에 존재하지 않습니다.")
        return
    
    gene_counts = df[gene_col].value_counts().head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=gene_counts.index, y=gene_counts.values, palette='viridis')
    plt.title(f'초록에서 언급 빈도가 높은 상위 {top_n} 유전자')
    plt.xlabel('유전자')
    plt.ylabel('빈도')
    plt.xticks(rotation=45)
    plt.text(0.5, 1.05, '바 플롯: 범주형 데이터의 빈도 비교 시각화',
             fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.show()

def main():
    # 파일 경로 설정
    abstracts_path = 'data/processed_abstracts.txt'
    rnaseq_path = 'data/processed_rnaseq_data.csv'
    clinical_path = 'data/processed_tcga_clinical_data.csv'
    
    # 1. 데이터 로드 및 전처리
    print("데이터 로드 및 전처리 중...")
    abstracts_df = load_and_preprocess_abstracts(abstracts_path)
    
    if abstracts_df.empty:
        print("초록 데이터프레임이 비어 있습니다. 분석을 중단합니다.")
        return
    
    # 2. 임베딩 생성
    print("임베딩 생성 중...")
    abstracts_df, embeddings = generate_embeddings(abstracts_df)
    
    if embeddings.size == 0:
        print("임베딩 생성에 실패했습니다. 분석을 중단합니다.")
        return
    
    # 3. 유사도 매트릭스 계산
    print("유사도 매트릭스 계산 중...")
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    if similarity_matrix.size == 0:
        print("유사도 매트릭스 계산에 실패했습니다.")
    
    # 4. RNA-seq 데이터 및 임상 데이터 로드
    print("RNA-seq 데이터 로드 중...")
    rnaseq_df = load_rnaseq_data(rnaseq_path)
    
    if rnaseq_df.empty:
        print("RNA-seq 데이터가 비어 있습니다. 분석을 중단합니다.")
        return
    
    print("임상 데이터 로드 중...")
    clinical_df = load_clinical_data(clinical_path)
    
    if clinical_df.empty:
        print("임상 데이터가 비어 있습니다. 분석을 중단합니다.")
        return
    
    # 5. RNA-seq 데이터와 임상 데이터 병합
    print("RNA-seq 데이터와 임상 데이터 병합 중...")
    merged_df = merge_rnaseq_clinical(rnaseq_df, clinical_df)
    
    if merged_df.empty:
        print("RNA-seq 데이터와 임상 데이터 병합에 실패했습니다. 분석을 중단합니다.")
        return
    
    print(f"병합된 데이터 형태: {merged_df.shape}")
    
    # 6. 차원 축소 (PCA 및 t-SNE)
    print("PCA 적용 중...")
    pca_result = apply_pca(embeddings)
    
    print("t-SNE 적용 중...")
    tsne_result = apply_tsne(embeddings)
    
    # 7. 시각화를 위한 가상의 라벨 추가 (실제 데이터에 맞게 수정 필요)
    # 예를 들어, 특정 유전자와 관련된 초록이라면 유전자 이름을 라벨로 사용할 수 있습니다.
    # 현재는 임의로 유전자 이름을 할당
    if not abstracts_df.empty:
        genes = ['BRCA1', 'TP53', 'EGFR', 'MYC', 'PTEN']
        abstracts_df['gene_name'] = np.random.choice(genes, size=len(abstracts_df))
    else:
        abstracts_df['gene_name'] = 'Unknown'
    
    # 8. 차원 축소 결과 시각화 (유전자별 색상 구분)
    print("차원 축소 결과 시각화 중...")
    visualize_dimension_reduction(abstracts_df, pca_result, tsne_result, labels='gene_name')
    
    # 9. 유전자 발현 수준의 히트맵 시각화
    print("유전자 발현 히트맵 시각화 중...")
    visualize_heatmap(rnaseq_df, top_n=50)
    
    # 10. 유전자 간 상관관계 히트맵 시각화
    print("유전자 상관관계 히트맵 시각화 중...")
    visualize_correlation_heatmap(rnaseq_df, top_n=30)
    
    # 11. 임상 변수와 유전자 발현 간의 관계 시각화
    print("임상 변수와 유전자 발현 간의 관계 시각화 중...")
    gene_of_interest = 'ENSG00000223972.5'  # 예시 유전자 ID
    if gene_of_interest not in rnaseq_df.columns:
        print(f"'{gene_of_interest}' 유전자가 데이터에 존재하지 않습니다. 다른 유전자로 대체합니다.")
        gene_of_interest = rnaseq_df.columns[0]  # 첫 번째 유전자로 대체
    clinical_var = 'days_to_death'  # 예시 임상 변수
    visualize_relationship(merged_df, clinical_var, gene_of_interest)
    
    # 12. 그룹 간 유전자 발현 수준 비교 시각화
    print("그룹 간 유전자 발현 수준 비교 시각화 중...")
    group_var = 'disease_code'  # 예시 그룹 변수
    visualize_boxplot(merged_df, group_var, gene_of_interest)
    
    # 13. 유전자 발현 빈도 시각화
    print("유전자 발현 빈도 시각화 중...")
    visualize_expression_frequency(rnaseq_df, top_n=20)
    
    # 14. 추가 시각화: 발현 빈도 (가상의 라벨 사용)
    print("발현 빈도 시각화 (가상의 라벨) 중...")
    visualize_gene_occurrence_frequency(abstracts_df, gene_col='gene_name', top_n=20)

# 코드 실행
if __name__ == "__main__":
    main()
