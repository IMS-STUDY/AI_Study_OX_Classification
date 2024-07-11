# 인공지능 7/8

- 발표/스터디
    
    CNN
    
    - 완전연결계층
    - 합성곱
    - 패딩
    - 풀링
    - conv3d : 비디오 데이터 +…
    - convd 더 알아보기
    
    [3D Convolution 완전 정복하기: Using PyTorch Conv3D](https://medium.com/@parkie0517/3d-convolution-완전-정복하기-using-pytorch-conv3d-4fab52c527d6)
    
    [Conv1D, Conv2D, Conv3D 차이](https://leeejihyun.tistory.com/37)
    
- 과제
    - [x]  acitvation function
    - ~~acitvation function란?~~
        - ~~목적~~
    - ~~종류, 특징~~
        - ~~sigmoid~~
        - ~~tanh~~
        - ~~relu~~
        - ~~softmax~~
    - [x]  k-fold
        - k-fold란?
        - k-fold 원리
    
    [실습]
    
    주피터에 코드 누적
    
    - [ ]  OX 데이터셋 MLP로 분류 진행하기
        - MLP로 진행시 k-fold적용 , 미적용 각각 정확도 산출
1. ~~데이터셋 불러오기~~
2. ~~전처리~~
3. MLP로 분류

# Activation function

- 정의
    
    활성화 함수는 노드의 개별 입력과 가중치를 기반으로 노드의 출력을 계산하는 비선형 함수임
    
    활성화 함수가  인 경우 몇 개의 노드만 사용하여 사소한 문제를 해결할 수 있음
    
    선형 함수는 입력 값에 대해 일정한 비율로 결과값이 변하는 함수임
    
    비선형 함수는 함수 그래프가 직선이 아닌 함수로, 입력 값에 대해 비선형적으로 출력 값을 결정
    
- 목적
    
    다음 layer로 어떤 값을 전달할지, 활성화, 비활성화를 결정함
    
- 종류, 특징
    - **sigmoid**
        ![%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%ED%95%A8%EC%88%981](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/fd9723e7-2234-4a79-97a2-906ed0ade89d)
        
        
        
        - 기존 계단 함수가 출력을 0과 1 이진 값만을 반환했다면 시그모이드는 계단 함수의 각진 부분을 매끄럽게 해서 0과 1 사이의 값을 반환계단 함수와는 다르게 연속적으로 변화.
        - 그래프가 S자 모양 또는 시그모이드 곡선의 특징을 갖는 수학 함수임
        
        ![e5b42d3fad3c41825a1493a0daa271523cbab01c](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/8daced96-c0fb-4141-a3d1-31f7c8187ed4)
        
        - 실수 입력값에 대해 정의되고 각 지점에서 음이 아닌 미분과 정확히 하나의 변곡점을 가짐. 미분 가능
        - 출력은 0과 1 사이
        - 이진 분류, 로지스틱 회귀나 인공신경망의 출력층 등에서 자주 사용 됨
        - 입력값이 너무 크거나 작으면 Vanishing Gradient(기울기 소멸) 문제 발생
        - Vanishing Gradient : 역전파 과정에서 0에 가까운 작은 기울기가 곱해지면 입력층으로 갈수록 기울기가 작아지는 현상이 발생. 입력층에 인접한 층에서의 가중치가 업데이트되지 않음.
        
        ![img](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/179eed18-bffe-48c0-82a4-6c872814ccfc)
        
        - 데이터가 이진, 범주형 이라면 시그모이드 사용을 권장함.
    - **tanh**
        
        ![%ED%95%98%EC%9D%B4%ED%8D%BC%EB%B3%BC%EB%A6%AD%ED%83%84%EC%A0%A0%ED%8A%B8](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/e2db7fbc-7bcd-42a8-9ebc-0a16add7866a)
        
        - 출력값이 -1과 1 사이. 양의 무한대로 갈수록 1에 수렴함.
        - 비선형성을 모델에 부여할 수 있음
        - 경사 하강 알고리즘의 수렴을 더 빠르게 만들 수 있음.
        - 시그모이드와는 다르게 중앙값이 0이며 미분 최댓값은 1이다. 이는 시그모이드를 보완한 함수임.
        - vanishing gradient문제를 완화시킴
        - 정규화 문제 등 다양하게 사용됨
        - 
    - **ReLU**
        
        ![%EB%A0%90%EB%A3%A8%ED%95%A8%EC%88%98](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/498642c6-a752-4d1d-9367-a721be556c38)
        
        - 음수를 입력하면 0을 출력, 양수를 입력하면 양수 그대로 출력함
        - 간단한 함수라서 시그모이드 보다 계산 효율성이 뛰어남.
        - Vanishing Gradient 문제를 해결함→ 양수 입력값에 대한 일정한 기울기를 가지기 때문
        - Saturation problem(포화 문제) 방지
        - Dying ReLU 발생→ 입력값이 음수면 기울기도 0이 된 뉴런을 회생하는 것이 매우 어려움
        
        ![리키렐루.png](%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A9%E1%86%BC%E1%84%8C%E1%85%B5%E1%84%82%E1%85%B3%E1%86%BC%207%208%20be618a2b171d48dca06aa7d6488ea876/%25EB%25A6%25AC%25ED%2582%25A4%25EB%25A0%2590%25EB%25A3%25A8.png)
        
        - 음수를 입력해도 기울기가 0이 되지 않음.
        - Dying ReLU 문제를 해결
        - PReLU→새로운 파라미터 α 를 추가해 x가 음수인 영역에서도 기울기를 학습함. a의 값을 사용자가 지정 가능
            
            ![0SQXAzL](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/e44242ac-514c-4ef7-8e28-8bf092c2e662)
            
        - ELU→ReLU의 장점을 포기하고 Dying ReLU문제를 해결함(연산 비용 발생)
            
            ![F7apORx](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/4b2ed2aa-b1cf-4d9c-888b-2495ec29c30b)
            
    - **softmax**
        
        ![%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/ba0c23b3-34b4-4b2a-bae6-12ed759868d4)
        
        - K개의 실수 값 벡터를 합이 1이 되는 K개의 실수 값 벡터로 바꾸는 함수
            
            선택지의 총 개수를 k라고 할 때, k차원의 벡터를 입력받아 각 클래스에 대한 확률을 추정함
            
        - 입력 값은 양수, 음수, 0 또는 1보다 클 수 있지만 소프트맥스는 이를 0과 1 사이의 값으로 변환하여 확률로 해석함
        - 소프트맥스는 점수를 정규화된 확률분포 로 변환하기 때문에 매우 유용함. 이러한 이유로 신경망 마지막 층에 추가하는 것이 일반적임
        - 시그모이드와 유사하지만 소프트맥스는 벡터에서 작동하고 시그모이드는 스칼라를 취함
        - 
        
- 참고자료
    
    [Softmax Function](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)
    
    [#13 Softmax Function](https://velog.io/@chiroya/13-Softmax-Function)
    
    [딥러닝-3.4. 활성화함수(5)-렐루 함수(ReLU)](https://gooopy.tistory.com/55)
    
    [hyperbolic tangent, tanh (하이퍼볼릭 탄젠트, 쌍곡 탄젠트)](https://wikidocs.net/152159)
    
    [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
    
    [06-06 비선형 활성화 함수(Activation function)](https://wikidocs.net/60683)
    
    [Activation function](https://en.wikipedia.org/wiki/Activation_function)
    

# K-fold

- 정의
    
    k개의 fold를 만들어 진행하는 교차 검증
    
    - 총 데이터 갯수가 적은 데이터셋에 대하여 정확도 향상이 가능함
    - Training, Validation, Test 세 개의 집단으로 분류하는 것보다 Training, Test로만 분류했을 때의 학습 데이터가 많아 이득임
    - 검증과 테스트에 데이터를 뺐기면 underfitting 등 성능이 미달되는 모델이 학습됨
- 원리
    - 데이터를 k개로 나누고 k개의 fold로 나눈다.
        - 하늘색 : Training Data(k-1개)
        - 주황색 : Test Data(1개)
    
    |  |  |  |  |  |
    | --- | --- | --- | --- | --- |
    
    |  |  |  |  |  |
    | --- | --- | --- | --- | --- |
    
    |  |  |  |  |  |
    | --- | --- | --- | --- | --- |
    
    |  |  |  |  |  |
    | --- | --- | --- | --- | --- |
    
    |  |  |  |  |  |
    | --- | --- | --- | --- | --- |
    - 모델을 생성하고 예측을 진행해 이에 대한 에러값을 추출한다.
    - 다음 fold에서는 Test Data셋을 바꾸어 진행하고 이전의 Test Data는 Training Data로 다시 활용해 예측을 진행한다.
    - k번 반복
    
- 참고자료

# [실습]

1. 패키지 import
2. 데이터셋 가져오기
- Dataset, DataLoader
- label = ‘O’, ‘X’
- train, test 데이터 비율은?
- __init__, __len__, __getitem__ 메소드?
1. MLP로 분류(k-fold적용, 미적용 결과 각각 산출)

[PyTorch 나만의 데이터셋을 만들고, 이를 ImageFolder로 불러오기](https://ndb796.tistory.com/373)

[03-07 커스텀 데이터셋(Custom Dataset)](https://wikidocs.net/57165)

1. 데이터셋 파일을 가져옴
- 압축한 파일을 업로드→ 아래 코드를 통해 압축 해제

```python
# Dataset 폴더에 압축 해제
!mkdir -p /content/OX_Dataset
!unzip -qq /content/OX_Dataset.zip -d /content/OX_Dataset
```

1. 필요한 라이브러리 import

```python
# 라이브러리 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.data import Dataset # Import the Dataset class
from PIL import Image 
```

1. 데이터를 가져와 텐서로 만듦??

```python
trans = transforms.Compose([transforms.Resize((224, 224)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trainset = datasets.ImageFolder(root='/content/OX_Dataset', transform=trans)
```

1. 레이블링 과정 + 데이터 셔플, 나눔

batch_size : 뭐지?

```python
# Create a list to store the modified data
modified_data = []

for i in range(len(trainset)):
  image, label = trainset[i]  # Get the image and label
  if i < 140:
    label = 0 
  else:
    label = 1
  modified_data.append((image, label))  # Append the modified data

# Create a new custom dataset from the modified data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Initialize classes and class_to_idx within the class definition
        self.classes = ['O', 'X']
        self.class_to_idx = {'O': 0, 'X': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create an instance of the custom dataset
modified_trainset = CustomDataset(modified_data)

# Use the modified_trainset for further processing
trainloader = torch.utils.data.DataLoader(modified_trainset, batch_size=10, shuffle=True)

# Access the attributes of the CustomDataset instance
print(modified_trainset.classes)
print(modified_trainset.class_to_idx)
```

1. 시각화

```python
# 시각화
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show

images, labels = next(iter(trainloader))
print(images.shape)
print(labels)
imshow(torchvision.utils.make_grid(images))
```

결과

```
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

```

```
torch.Size([10, 3, 224, 224])
tensor([1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
```


![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/9b198fe4-9f84-4297-90c6-d4f8c4f63086)
---

[(Pytorch) MLP로 Image Recognition 하기](https://gaussian37.github.io/dl-pytorch-Image-Recognition/)

1. Classifier 클래스 정의
2. loss function 설정
- 대표적으로 CrossEntropyLoss를 사용한다고 함
1. optimizer 설정
- Adam을 사용함
- 옵티마이저가 뭐지?
1. training과 validation의 loss와 accuracy를 저장할 리스트를 생성
2. 와떠헬