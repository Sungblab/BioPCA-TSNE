import requests
from bs4 import BeautifulSoup
import time

def fetch_pubmed_abstracts(query, max_results=20):
    """
    PubMed에서 특정 쿼리에 해당하는 논문 초록을 수집합니다.
    :param query: 검색할 유전자 이름이나 키워드 (예: 'BRCA1 gene')
    :param max_results: 각 유전자에 대해 검색할 최대 결과 수
    :return: 초록 리스트
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml"
    }

    # 논문 ID 검색
    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, "xml")
    ids = [id_tag.text for id_tag in soup.find_all("Id")]

    # 각 논문의 초록 가져오기
    abstracts = []
    for pmid in ids:
        abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=text&rettype=abstract"
        
        abstract_response = requests.get(abstract_url)
        if abstract_response.status_code == 200:
            abstracts.append({
                "pmid": pmid,
                "abstract": abstract_response.text
            })
            print(f"Fetched abstract for PMID {pmid} on query '{query}'")
        else:
            print(f"Failed to retrieve abstract for PMID {pmid} on query '{query}'")
        
        # 요청 간 짧은 대기시간 추가 (API 제한 방지)
        time.sleep(0.5)
        
    return abstracts

# 유전자 목록 설정 (여기에 다양한 유전자 이름을 추가할 수 있습니다)
gene_queries = ["BRCA1 gene", "TP53 gene", "EGFR gene", "MYC gene", "KRAS gene"]

# 수집할 초록 수
max_results_per_gene = 20

# 모든 유전자에 대해 논문 초록 수집
all_abstracts = []
for gene in gene_queries:
    abstracts = fetch_pubmed_abstracts(gene, max_results=max_results_per_gene)
    all_abstracts.extend(abstracts)

# 수집한 데이터를 파일에 저장
with open("multi_gene_pubmed_abstracts.txt", "w", encoding="utf-8") as f:
    for abstract in all_abstracts:
        f.write(f"PMID: {abstract['pmid']}\n")
        f.write(f"Abstract: {abstract['abstract']}\n\n")

print("Abstracts saved to multi_gene_pubmed_abstracts.txt")
