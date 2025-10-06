# <벡터들 간 코사인 유사도 계산>

import numpy as np
from numpy import dot
from numpy.linalg import norm

# 코사인 유사도를 계산하는 함수를 정의

def cos_sim(A,B):
    return dot(A,B)/(norm(A)*norm(B))

# 벡터들 간의 코사인 유사도를 출력
vec1 = np.array([0,1,1,1])
vec2 = np.array([1,0,2,1])
vec3 = np.array([2,0,4,2])

print('벡터1과 벡터2의 유사도:',cos_sim(vec1,vec2))
print('벡터1과 벡터3의 유사도:',cos_sim(vec1,vec3))
print('벡터2과 벡터3의 유사도:',cos_sim(vec2,vec3))