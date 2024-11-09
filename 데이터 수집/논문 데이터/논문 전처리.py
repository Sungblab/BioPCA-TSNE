import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 초기화
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 불필요한 정보 제거 (PMID, 저널 정보 등)
    text = re.sub(r"PMID: \d+|doi: \S+|Author information:.*|eCollection.*", "", text)
    
    # 특수 문자 및 숫자 제거
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # 소문자 변환
    text = text.lower()
    
    # 토큰화 및 불용어 제거
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    
    # 표제어 추출
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 전처리된 텍스트 반환
    return " ".join(words)

# 입력 파일 경로와 출력 파일 경로 설정
input_file = "multi_gene_pubmed_abstracts.txt"
output_file = "processed_abstracts.txt"

# 파일 읽기, 전처리 및 저장
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    current_abstract = ""
    for line in infile:
        # 초록 데이터가 끝나면 전처리하여 파일에 저장
        if line.strip() == "" and current_abstract:
            processed_text = preprocess_text(current_abstract)
            outfile.write(processed_text + "\n\n")
            current_abstract = ""
        else:
            current_abstract += line

    # 마지막 초록 전처리 후 저장 (파일 끝 처리)
    if current_abstract:
        processed_text = preprocess_text(current_abstract)
        outfile.write(processed_text + "\n\n")

print(f"Processed abstracts saved to {output_file}")
