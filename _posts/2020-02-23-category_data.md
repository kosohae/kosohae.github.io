---
layout: post
title: 범주형 데이터 모델링
subtitle: 카테고리 데이터 속성 및 범주형 데이터 다루기
bigimg: /img/path.jpg
category: statistics
tags: [statistics]
comments: true
---

# 범주형 데이터

분류 모델을 만들어 보려다가 사용 데이터 중에 한 column이 범주형(category) 형태임을 발견했다.
모델의 경우, 수치형 데이터들이 들어가서 affine 계산 및 최적화를 진행하는데 위와 같은 데이터가 범주형으로 그냥 들어가게 된다면 수치계산을 할 수 없다고 생각했다.

- 숫자로 변환한다.
두 가지 방법을 참고했다. 1) 더미변수화 2) 카테고리 임베딩
1) 더미변수
1 - K 까지 값을 가지는 범주형 변수에 대해 0,1로 구성된 K개의 더미변수 벡터로 표현한다. 
1-1) full-rank 방식
A,B,C에 대해 [1,0,0] [0,1,0] [0,0,1] 로 표현하는 것. 이 경우, 카테고리 수가 많을 경우 높은 차원의 더미변수 벡터를 갖게 될 수 있다.
1-2) recude-rank 방식
특정한 하나의 범주값을 기준값으로 놓고, 기준값에 대응하는 더미변수의 가중치를 항상 1로 놓음
[1,0,0] [1,1,0] [1,0,1] 이런식으로 예를 들 수 있을 듯 하다.

- 도구
만약 문자열 데이터가 범주형으로 표현되어 있는 경우 : patsy 패키지의 dmatrix() 함수
일반 카테고리에 맞춰 범주형 더미 변수로 변형 : pandas get_dummies()

더미변수의 경우, 데이터 사이언스 스쿨의 자료를 참고했다.
(https://datascienceschool.net/view-notebook/953eb363588c421d9162330c6c7df901/)


2) 임베딩
범주형 자료를 연속형 벡터 형태로 변환시키는 것.
각 벡터에 가장 알맞는 값을 할당하게 된다. 이 과정에서 각 범주(category)는 각각의 특징을 가장 잘 표현할 수 있는 형태의 벡터값.

* 추천 시스템이나 군집분석에 활용
* 범주관계를 시각화 활용
* embedding에 활용할 벡터의 길이를 정한다. 특별한 기준은 없지만 rule of thumbs는 아래와 같다.
- embedding size = min(50, number of categories/2)


참고 : https://github.com/yerimlim/TodayILearned/wiki/2.-Embedding%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80

- 2020/2/27 update : entity embedding for categorical variables
범주형 카테고리에 대해 신경망 알고리즘을 적용하는 내용이 있어서 간단히 논문을 보고 torch로 구현해보았다.

```Python
class Embedding(nn.Module):
    def __init__(self, n):
        super().__init__()
        """ Embedding model for entity 
            Args : n(int) output shape
        """
        self.n = n
        self.embed = nn.Embedding(1001,n)
        
    def forward(self, x):
        outputs = self.embed(x)
        #outputs = outputs.reshape(self.n,)
        return outputs
        #self.fiber = self.get_fiber(torch.LongTensor((self.values.iloc[:,2].values, ))) 
        #self.tmp_x = np.append(self.fiber.squeeze(0).detach().numpy(), self.tmp_x, axis=1) # float
    
        #self.tmp_x = self.normalize(self.tmp_x)

# apply
class PandasDataset(Dataset):
    def __init__(self, data_path):
        super(PandasDataset, self).__init__()
        self.values = pd.read_csv(data_path)
        self.X, self.Y = self.values.iloc[:,3:], self.values.iloc[:,1] # type
        self.tmp_x , self.tmp_y = self.X.values, self.Y.values
        self.fiber = self.get_fiber(torch.LongTensor((self.values.iloc[:,2].values, ))) 
        self.tmp_x = np.append(self.fiber.squeeze(0).detach().numpy(), self.tmp_x, axis=1) 
    
    def get_fiber(self, x):
        embed_model = Embedding(10)
        fiber = embed_model(x)
        return fiber
```

- Embedding layer를 만들어서 인덱스에 맞게 카테고리 범주를 벡터로 변환한다.
- 다만 처음 벡터가 어떤 속성이 반영된 벡터값은 아니고 단순분포에 의해 생성된 값이다. (Pytorch Embedding weights 기본생성 과정)