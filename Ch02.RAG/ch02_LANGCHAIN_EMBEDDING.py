import os
from dotenv import load_dotenv
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
query_result = embeddings.embed_query('저는 배가 고파요')
# print(query_result)

# <오픈AI 임베딩을 활용한 문장들 간 코사인 유사도 계산>

data = [
    '주식 시장이 급등했어요',
    '시장 물가가 올랐어요',
    '전통 시장에는 다양한 물품들을 팔아요',
    '부동산 시장이 점점 더 복잡해지고 있어요',
    '저는 빠른 비트를 좋아해요',
    '최근 비트코인 가격이 많이 변동했어요'
]

df =pd.DataFrame(data, columns=['text'])
# print(df)

# 텍스트를 임베딩 벡터를 변환하는 함수 정의

def get_embedding(text):
    return embeddings.embed_query(text)

# DataFrame의 각 행에 대해 'text' 열의 내용을 임베딩 벡터로 변환
df['embedding']= df.apply(lambda row: get_embedding(
    row.text,
),axis=1)

print(df)