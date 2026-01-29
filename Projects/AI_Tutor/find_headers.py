from langchain_community.document_loaders import PyPDFLoader
import os

DATA_DIR = "/home/sssss_mmm/langchain/Projects/AI_Tutor/Data"
sample_file = "1. 2024년1회_정보처리기사필기기출문제.pdf"
file_path = os.path.join(DATA_DIR, sample_file)

if os.path.exists(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    found_headers = []
    
    for i, page in enumerate(pages):
        lines = page.page_content.split('\n')
        for line in lines:
            if "과목" in line:
                found_headers.append(f"Page {i+1}: {line.strip()}")
                
    print(f"Found {len(found_headers)} headers containing '과목':")
    for h in found_headers[:20]:
        print(h)
