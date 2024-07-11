# 인공지능 2-1과제

2주-1

주피터에 코드 누적

- [x]  OX_Classification
- MLP로 분류 진행하기
    - ~~OX_Dataset 불러오기~~
    - ~~OX_image 전처리~~
    - MLP로 진행시 k-fold적용 , 미적용 각각 정확도 산출
- ~~acitvation function~~
    - ~~acitvation function란?~~
        - ~~목적~~
    - ~~종류, 특징~~
        - ~~sigmoid~~
        - ~~tanh~~
        - ~~relu~~
        - ~~softmax~~
- ~~k-fold~~
    - ~~k-fold란?~~
    - ~~k-fold 원리~~
    

## 1. MLP로 분류 진행하기

### 1) OX_dataset 불러오기

```python
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

path = r"C:\Users\USER\Desktop\Dataset"
def findFiles(path): return glob.glob(path)
#데이터 불러오기
```

➡ 데이터를 불러오는 과정이다. 받은 데이터 파일의 디렉토리 주소(절대경로)로 접근 하였음을 확인할 수 있다.

```python

#하이퍼 파라미터 설정
training_epochs = 10
batch_size = 32

learning_rate = 0.001

input_size = 90000    # 300x300 image       # 고정된 값 (이미지 크기)
hidden_size = 1000   # 임의의 값           # 임의의 값 (hidden layer의 노드 개수)
output_size = 2    # OX               # 고정된 값 (분류할 클래스 개수)
```

➡ 하이퍼 파라미터를 설정하는 코드이다. 

기본적인 epoch와 batch size, 그다음 input size와 hidden size, output size를 정의해주는 과정을 거친다.

### 2) OX_image 전처리

```python
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

image_list1 = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.png')]
image_list2 = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.jpg')]

image_list = image_list1+image_list2
transform = transforms.ToTensor()

#image = Image.open(image_list[0]).convert("L")
#image = transform(image)
dataset = [transform(Image.open(idx).convert("L")) for idx in image_list]

print(f"dataset 배열 크기 : {len(dataset)}")
print(f"첫번째 원소값 사이즈: {dataset[0].size()}")
```

```
dataset 배열 크기 : 280
첫번째 원소값 사이즈: torch.Size([1, 300, 300])
```

➡ path에 있는(dataset) png파일과 jpg파일을 꺼내 리스트에 담고 각각의 두 리스트를 하나의 image_list에 합친다. 그 다음 transforms.ToTensor()를 통해 image_list의 모든 데이터를  **"Channel x Height x Width" 구조**로 바꿔준다. 

➡ 배열 크기는 280개 (O : 140개, X : 140개)가 된 것을 확인할 수 있다.  

<aside>
💡 **이후 사이즈를 바꾸는 과정과 라벨링을 추가함**

</aside>

```python

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import cv2
import numpy as np
from matplotlib import pyplot as plt

image_list1 = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.png')]
image_list2 = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.jpg')]

image_list = image_list1+image_list2
transform = transforms.ToTensor()

#image = Image.open(image_list[0]).convert("L")
#image = transform(image)

dataset = []
labels = []

for idx in image_list:
    image = cv2.imread(idx, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (50,50))
    image = Image.fromarray(image)
    image = transform(image)
    dataset.append(image)

    filename = os.path.basename(idx).lower()
    if 'o' in filename:
        labels.append(0) #결과 레이블 'o'
    elif 'x' in filename:
        labels.append(1) #결과 레이블 'x'

print(f"dataset 배열 크기 : {len(dataset)}")
print(f"첫번째 원소값 사이즈: {dataset[0].size()}")
```

```
dataset 배열 크기 : 280
첫번째 원소값 사이즈: torch.Size([1, 50, 50])
```

➡ 사이즈 크기를 50*50 픽셀의 이미지로 변환하였다. 그리고 이미지의 파일명을 통해서 데이터의 라벨링 배열을 만들어주었다.

```python
# 리스트를 텐서로 변환
labels = torch.tensor(labels, dtype=torch.long)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(list(zip(dataset, labels)), [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False
```

➡ 그다음 8:2 비율로 train 데이터와 test 데이터를 나눠 담는다. 그리고 `DataLoader`를 이용해 train 데이터와 test 데이터를 배치 단위로 로드할 수 있도록 합니다. batch size는 위에서 정의한 32이라는 값이다.

### 3) MLP로 진행시 k-fold적용 , 미적용 각각 정확도 산출

- 미적용의 경우

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(),  # 50*50 이미지를 2500*1 벡터로 변환
            nn.Linear(in_features=2500, out_features=1250, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1250, out_features=625, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=625, out_features=315, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=315, out_features=150, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=70, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=70, out_features=2, bias=True)
        )

    def forward(self, x):
        return self.sequential(x)

model = MLP()
criterion = nn.CrossEntropyLoss() #손실함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #옵티마이저
#learning rate 0.1 -> 0.0001로 변경
```

➡  MLP 모델을 만들어주고 손실함수와 옵티마이저를 정의 해준다. 옵티마이저는 유명한 Adam을 사용했고, 손실함수도 2진 분류에 적합한 CrossEntropy를 사용했다.

- 모델 학습

```python
for epoch in range(100): #10->100으로 증가
    avg_cost = 0
    total_batch = len(train_loader)
    
    for images, labels in train_loader:
        optimizer.zero_grad()

        hypothesis = model(images)

        cost = criterion(hypothesis, labels)
        cost.backward()
        
        optimizer.step()
        avg_cost += cost / total_batch

    print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))

```

```
Epoch: 0001 cost = 0.693798721
Epoch: 0002 cost = 0.692188799
Epoch: 0003 cost = 0.691086352
Epoch: 0004 cost = 0.689724028
Epoch: 0005 cost = 0.686602116
Epoch: 0006 cost = 0.691653848
Epoch: 0007 cost = 0.683707118
Epoch: 0008 cost = 0.679791152
Epoch: 0009 cost = 0.676595092
Epoch: 0010 cost = 0.671693802
...
Epoch: 0097 cost = 0.148798212
Epoch: 0098 cost = 0.160334587
Epoch: 0099 cost = 0.080024049
Epoch: 0100 cost = 0.068397321
```

➡ 그 다음 모델 학습을 진행한다. epoch값은 기존에 10으로 설정해놨지만 100으로 늘려놨으며, 위에 옵티마이저의 learning rate도 0.0001 변경했다. 

print로 출력을 해보면 cost(손실)이 꾸준히 줄어드는 것을 확인할 수 있다.

- 모델 평가

```python
import random
import matplotlib.pyplot as plt

# 테스트 데이터를 사용하여 모델을 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않음
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# 테스트 데이터에서 무작위로 하나를 뽑아서 예측
r = random.randint(0, len(test_dataset) - 1)
single_image, single_label = test_dataset[r]
single_image = single_image.unsqueeze(0)  # 배치 차원을 추가
with torch.no_grad():
    single_prediction = model(single_image)
    predicted_label = torch.argmax(single_prediction, 1).item()

print(f'Label: {single_label.item()}')
print(f'Prediction: {predicted_label}')

# 이미지 출력
plt.imshow(single_image.squeeze().numpy(), cmap="Greys", interpolation="nearest")
plt.show()
```

```python
Accuracy of the model on the test images: 75.00%
Label: 1
Prediction: 1
```

![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/6c74f396-c56b-4472-835c-189ce2eec6ae)

➡️ 최종적으로 75% 정도의 정확도가 나오는 것을 확인할 수 있다. 

---

## 2. Activation function

### 1) activation function란?

= 활성화 함수라는 뜻으로, 노드의 출력을 얻는 데 사용하는 단순한 사물 함수를 의미한다. 전달함수의 의미로 사용되기도 한다.

- 활성화 함수를 사용하는 이유(목적)

= 신경망의 출력을 예/아니오로 결정하는데 사용된다. 혹은 결과값을 0~1또는 -1~1 등으로 매핑하는 역할을 한다. 이는 함수에 따라 결과값이 다르지만 결론적으로는 출력을 제어하는데 사용된다고 생각하면 된다.

### 2) 종류, 특징

활성화 함수는 기본적으로 2가지 유형으로 나눌 수 있다. 

→ **선형 활성화 함수**

![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/085ff007-b284-4f21-9163-a828b9fb99ac)

방정식 :  f(x) = x

범위 : ( -∞, +∞ )

특징 : 해당 활성화 함수로는 신경망에 입력되는 복잡한 데이터나 다양한 매개변수 처리는 힘들다.  

→ **비선형 활성화 함수**

![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/eaff28e9-91b0-40a0-b4ee-4b27a0b4586b)

: 비선형 활성화 함수는 연속적이고 불규칙한 데이터에 사용될 수 있으며, 다양한 데이터에 맞게 모델을 일반화하거나 적응하고 출력을 구별하는 것이 쉬워진다.

아래에 이어질 활성화 함수도 이러한 비선형 활성화 함수들이다.

- **sigmoid**

![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/acc83387-8269-4a2f-873e-1298e8b333e1)

: 시그모이드 함수의 곡선은 S자 모양이다. 

```python

# sigmoid 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid 함수의 도함수
def sigmoid_derivative(x):
    sigmoid_x = 1 / (1 + np.exp(-x))
    derivative = sigmoid_x * (1 - sigmoid_x)
    return derivative
```

시그모이드 함수를 사용하는 주된 이유는 **0~1 사이에 존재하기 때문이다.** 따라서 출력을 확률로 예측하기 위해서 자주 사용되며, **확률은 0~1의 범위 사이에 존재하기 때문에 시그모이드가 적합**하다고 볼 수 있다.

- tanh

![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/183e8d4c-9e9d-4570-bdba-72e3de9621d3)

: tanh는 시그모이드 함수와 비슷하지만 범위에서 차이점이 있다. 

tanh 함수의 출력 범위는 (-1, 1)로, 시그모이드처럼 S자 곡선 형태며 음수 입력이 강하게 음수로 매핑되고 0 입력은 0에 가깝게 매핑할 수 있다는 점이다. 

```python
# tanh 활성화 함수
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# tanh 함수의 도함수
def tanh_derivative(x):
    tanh_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    derivative = 1 - tanh_x**2
    return derivative
```

tanh 함수는 주로 두 클래스 간의 분류에 사용된다.

- relu

ReLU는 현재 가장 많이 사용되는 활성화 함수로, 거의 모든 합성곱 신경망이나 딥러닝에 사용된다. 

![Untitled 5](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/c5f0bb8c-19c6-40d8-83f9-a426d06150eb)

Sigmoid vs ReLU

ReLU 함수는 0이하의 음수값은 전부 0으로 처리되고 양수값은 그대로 출력되는 특징을 가지고 있다.
→ 범위 : (0 , +∞ )

```python
# ReLU 활성화 함수
def relu(X):
    return np.maximum(0, X)

# ReLU 함수의 도함수
def relu_derivative(X):
    return np.where(X > 0, 1, 0)
```

하지만 모든 음수값이 즉시 0이 되어 모델이 데이터를 적절히 맞추거나 학습하는 능력은 감소한다. 즉, ReLU 활성화 함수에 주어진 음수 입력은 그래프에서 값을 즉시 0으로 바꾸고, 적절한 매핑이 이루어지지않아 결과 그래프에 영향을 미친다.

- softmax

= softmax 함수는 확률처럼 모든 출력 값을 더했을 때 1이 총합이라는 특징을 가지고 있다.

softmax 함수에 입력값을 넣으면, 그 값들을 모두 0과 1사이의 값으로 정규화해주는데, 이는 각 확률이 마이너스 값을 가지지않고 더했을 때 총합이 1이 되는 것과 매우 흡사하다.

→ softmax함수는 시그모이드 함수에서 입력값이 n개로 늘어난 것과 같다.

```python
# softmax 활성화 함수
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

# softmax 함수의 도함수
def softmax_derivative(X):
    e_x = np.exp(x - np.max(x))
    softmax_x = e_x / np.sum(e_x, axis=0)
    derivative = softmax_x * (1 - softmax_x)
    return derivative
```

: 정리하자면 시그모이드 함수는 입력값이 하나일 때 사용되는 함수고, 시그모이드 함수를 입력값이 여러개일 때도 사용할 수 있게 일반화한 것이 softmax함수이다. 

> https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
> 

> https://financial-engineering.medium.com/ml-softmax-소프트맥스-함수-2f4740141bfe
> 

---

## 3. k-fold

<aside>
💡 나와임….마? 마! 나 와 임

</aside>

### 1) k-fold란?

= **k-fold cross validation**란 ****학습용/평가용 데이터 세트를 나누는 방법론 중 하나를 의미한다. 

먼저 교차 검증에 대해 먼저 알아보자면 쉽게 말해 데이터를 여러 번 반복해서 나누고 여러 모델을 학습하여 성능을 평가하는 방법이다. 

이렇게 하는 이유는 데이터를 학습용/평가용 데이터 세트로 여러번 나눈 것의 평균적인 성능을 계산하면 한 번 나누어서 학습하는 것에 비해 일반화된 성능을 얻을 수 있기 때문이다. 

![cv](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/66228adf-0112-4788-9040-f8114c8df787)

:  k-fold 교차 검증은 데이터를 k개로 분할한 뒤,  k-1개를 학습용 데이터 세트로, 1개를 평가용 데이터 세트로 사용하는데 이를 k번 반복하여 k개의 성능 지표를 얻어내는 방법이다.

### 2) k-fold 원리

: k-fold는 보통 5또는 10을 사용하지만, 다른 값을 사용할 수도 있다.  

➡️ k=5인 경우를 예로 들자면, 데이터를 폴드(fold)라고 하는 비슷한 크기의 부분 집합 5개로 나눈다. 

첫 번째 모델은 첫 번째 fold를 평가용 데이터 셋으로 사용하고, 두 번째부터 다섯 번째까지의 폴드(4개의 폴드)를 학습용 데이터셋으로 사용한다.

그 다음 모델은 두 번째 폴드를 평가용으로 사용하고, 1, 3, 4, 5 폴드를 학습용 데이터셋으로 사용한다. 이후 나머지도 같은 과정을 진행하여 총 5개의 정확도 값을 얻는다.

- Python 구현 방법

: k-fold cross-validation을 python을 통해 구현하는 방법을 알아보자. 데이터는 iris 데이터를, 라이브러리로는 scikit-learn을 활용한다. 다양한 방법을 통해 cross-validation을 수행할 수 있지만, 여기서는 `sklearn.model_selection.cross_val_score`를 활용한다.

```python
# iris 데이터 불러오기
iris= load_iris()

# 로지스틱 회귀 인스턴스 생성
logreg = LogisticRegression()

# k=5로 k-fold cross validation 수행
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("교차 검증 점수: ", scores)
```

> https://incodom.kr/k-겹_교차_검증
> 

---