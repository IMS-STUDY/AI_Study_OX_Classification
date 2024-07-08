# 인공지능 1-2과제

- ~~데이터셋 O-X 인당 20개씩 그리기 ( O 20개, X 20개 , 300x300) 붓 픽셀 굵으면 좋음~~~
- ~~CNN 이론~~
    - ~~완전 연결 계층~~
        - ~~완전 연결 계층이란?~~
        - ~~문제점~~
    - ~~합성곱~~
        - ~~합성곱이란?~~
    - ~~padding~~
        - ~~padding 이란?~~
        - ~~padding 종류~~
    - ~~pooling~~
        - ~~사용이유~~
        - ~~특징 및 장단점~~
    - ~~conv(n)d [ n = 1, 2 ]~~
        - ~~입력데이터의 차원, kernel~~
- ~~pytorch에서 MLP 구현하는법 (이론)~~

## 1. CNN 이론

### 1) 완전 연결 계층

- 완전 연결 계층이란?

= 한 층(layer)의 모든 뉴런이 그 다음 층의 모든 뉴런과 연결된 상태를 의미한다. **1차원 배열의 형태로 평탄화된 행렬을 통해 이미지를 분류하는데 사용되는 계층**이다. 

- 문제점

= 한계점으로는 3차원 컬러 이미지의 정보 손실이 있다. 컬러가 포함된 3차원 이미지를 1차원으로 평탄화하면 공간 정보가 손실될 수 밖에 없으며, 이럴 경우 정보 부족으로 인해 이미지를 분류하는 데 한계가 생길 수 밖에 없다.

### 2) 합성곱

- 합성곱이란?

→ CNN의 합성곱층에서 사용되는 연산으로, 이 합성곱 연산을 통해서 **이미지의 특징을 추출**하는 역할을 한다.

합성곱의 연산은 **커널 또는 필터**라고 하는 n*m 크기의 행렬로 높이*너비 크기의 이미지를 처음부터 끝까지 겹치며 훑으면서 n*m 크기의 겹쳐지는 부분의 각 이미지와 커널의 원소 값을 곱해서 모두 더한 값을 출력으로 하는 것을 말한다. 이때, 이미지의 가장 왼쪽 위부터 가장 오른쪽 아래까지 순차적으로 훑는 과정을 거친다.

아래는 그림과 예제를 통해 이해해보도록 하자.

![https://velog.velcdn.com/images/bluewing0303/post/505efb3f-e0a6-46fc-9d9d-0b48c777d8ea/image.png](https://velog.velcdn.com/images/bluewing0303/post/505efb3f-e0a6-46fc-9d9d-0b48c777d8ea/image.png)

커널(필터)는 일반적으로 3*3 혹은 5*5를 사용한다.

합성곱은 이러한 계산과정을 거친다.

이 과정을 코드로 구현해보면 이러한 코드를 작성할 수 있다.

```python
image_height, image_width = image.shape #이미지의 크기
filter_height, filter_width = filter.shape #필터(커널)의 크기

#결과값을 담을 특징맵(feature map)의 크기 지정
output_image = np.zeros((image_height - filter_height +1, image_width - filter_width +1))

#for문을 돌며 합성곱의 결과를 output_image에 담는다.
for i in range(image_height - filter_height + 1):
	for j in range(image_width - filter_width + 1):
		output_image[i, j] = np.sum(image[i:i+filter_height, j:j+filter_width] * filter)
```

위와 같이 입력으로부터 커널을 사용하여 합성곱 연산을 통해 나온 결과를 특성 맵(feature map)이라고 한다.

여기에 추가로 입력에 있어서 커널이 이동하는 step을 설정할 수 있는데 이 step을 **스트라이드(stride)**라고 한다.

![https://velog.velcdn.com/images/bluewing0303/post/5143c8e9-03cb-4c3f-baf6-fe2475ae0efa/image.png](https://velog.velcdn.com/images/bluewing0303/post/5143c8e9-03cb-4c3f-baf6-fe2475ae0efa/image.png)

스트라이드를 지정한 입력과 커널의 특징맵 추출 과정

스트라이드가 있는 경우또한 코드로 작성해보겠다.

```python
stride = 2 #stride가 2인 경우
image_height, image_width = image.shape #이미지의 크기
filter_height, filter_width = filter.shape #필터(커널)의 크기

#stride가 존재하기 때문에 나누기를 통해 feature map의 크기를 계산.
output_height = (image_height - filter_height) // stride + 1
output_width = (image_width - filter_width) // stride + 1

output_image = np.zeros((output_height, output_width)) #특징맵 크기 설정

#range(n,m,k) -> n부터 m-1까지 k만큼의 step을 가짐.
for i in range(0, image_height - filter_height + 1, stride):
	for j in range(0, image_width - filter_width + 1, stride):
		#i와 j의 값이 stride 배 만큼 커지기 때문에 그만큼 값을 나눠줘야함.
		output_image[i // stride, j // stride] = np.sum(image[i:i+filter_height, j:j+filter_width] * filter)
```

- 가중치와 편향

이 합성곱에서 필터(커널)은 결국 MLP의 가중치를 의미하는 것과 같다.

![https://velog.velcdn.com/images/bluewing0303/post/32830fcd-13f7-4d96-af97-34b019dd81bb/image.png](https://velog.velcdn.com/images/bluewing0303/post/32830fcd-13f7-4d96-af97-34b019dd81bb/image.png)

이러한 특징은 CNN에서는 더 적은 가중치 수로 공간적 구조 정보를 보존한다는 특징이 있다.

추가로, 편향을 구현하기 위해서는 합성곱을 진행하고 나온 결과값에 편향 값을 더해주기만 하면된다.

![https://velog.velcdn.com/images/bluewing0303/post/6ed0e821-5f6d-475e-857a-8f11805f753b/image.png](https://velog.velcdn.com/images/bluewing0303/post/6ed0e821-5f6d-475e-857a-8f11805f753b/image.png)

편향은 하나의 값만 존재.

### 3) Padding

- Padding이란?

= 패딩은 합성곱 연산 이후에도 특성 맺의 크기가 입력의 크기와 동일하게 유지되도록 하기위해 사용된다.

![https://velog.velcdn.com/images/bluewing0303/post/8ffd4764-10d5-4a62-adf5-2f956827edf2/image.png](https://velog.velcdn.com/images/bluewing0303/post/8ffd4764-10d5-4a62-adf5-2f956827edf2/image.png)

가장 간단한 zero padding

패딩은 합성곱 연산을 하기 전에 입력의 가장자리에 지정된 개수의 폭만큼 행과 열을 추가해주는 것을 의미한다. 지정된 개수의 폭만큼 테두리를 추가하고, 주로 값을 0으로 채우는 제로 패딩을 사용한다.

- Padding 종류
    - valid padding
    
    = 패딩을 추가하지 않은 형태는 엄밀히 말하면 밸리드 패딩을 적용한 것이다. 이 밸리드 패딩을 적용한 필터는 항상 입력 사이즈보다 작은 특징 맵이 나온다. 
    
    하지만 밸리드 패딩은 합성곱 연산시 입력 데이터의 모든 원소가 같은 비율로 사용되지 않는 문제가 있다. 
    
    - Full padding
    
    = 풀 패딩은 입력 데이터의 모든 원소가 합성곱 연산에 같은 비율로 참여하도록 하는 패딩 방식을 풀 패딩이라고 한다. 
    
    이때, (x,x) 크기의 필터가 있을 때, 풀 패딩의 스트라이드 값은 x-1이 된다. 그러므로 풀패딩을 적용한 입력데이터 크기는 2배가 된다.
    
    - Same padding
    
    = 세임 패딩은 출력 크기를 입력 크기와 동일하게 유지시킨다. (x, x) 사이즈 필터가 있을 때, 세임 패딩의 스트라이드 값은 (x-1)/2가 된다. 
    
    이렇기 때문에 세임 패딩을 half padding이라 하기도 한다.
    

### 4) Pooling

- 풀링(Pooling)이란?

= 일반적으로 CNN에서는 합성곱 층 다음에는 풀링 층을 추가하는 것이 일반적이다. 풀링층에서는 특성 맵은 다운샘플링하여 특성 맵의 크기를 줄이는 풀링 연산을 하는 층이다.

풀링 연산에는 일반적으로 최대 풀링과 평균 풀링이 사용된다. 

![Untitled](%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A9%E1%86%BC%E1%84%8C%E1%85%B5%E1%84%82%E1%85%B3%E1%86%BC%201-2%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%2095941e0cd9454495838d40f289b5aa74/Untitled.png)

풀링 연산에서도 합성곱 연산과 마찬가지로 커널과 스트라이드의 개념을 가진다. 위의 그림은 스트라이드가 2일 때, 2*2 크기 커널로 맥스 풀링 연산을 했을 때 특성맵이 절반 크기로 다운 샘플링 되는 것을 보여준다. 

- Max Pooling : 커널과 겹치는 영역 안에서 최대값을 추출하는 방식
- Measure Pooling : 커널과 겹치는 영역 안에서 평균값을 추출하는 방식

풀링 연산은 커널과 스트라이드 개념이 존재한다는 점에서 합성곱 연산과 유사하지만, 합성곱 연산과의 차이점은 **학습해야할 가중치가 없으며, 연산 후에 채널 수가 변하지 않는다는 점**이다.

### 5) conv(n)d [ n = 1, 2 ]

- convn : n차원 컨볼루션.

→ C = convn(A, B) 는 배열 A와 B의 N차원 컨벌루션을 반환한다. 

→  C = convn(A, B) 는 배열 A와 B을 `shape`에 따라 컨벌루션을 반환한다. 예를 들어, `C = convn(A,B,'same')`은 크기가 `A`와 동일한, 컨벌루션을 반환한다.

- 입력데이터의 차원, kernel
    
    ![Untitled](%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A9%E1%86%BC%E1%84%8C%E1%85%B5%E1%84%82%E1%85%B3%E1%86%BC%201-2%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%2095941e0cd9454495838d40f289b5aa74/Untitled%201.png)
    

= 커널은 입력데이터가 고차원이거나 내적 연산량에 문제가 있는 경우, 사용되는 개념으로, feature space 계산에서 Φ가 어떤 함수인지는 몰라도 Φ의 내적을 커널을 통해 계산할 수 있게 한다. 

[+) 추가글 → 커널의 조건 및 다양한 커널 함수의 설명](https://sonsnotation.blogspot.com/2020/11/11-1-kernel.html)

## 2. pytorch에서 MLP 구현하는법(이론)

- 데이터셋

= 데이터 셋으로는 보스턴 집값 데이터, MNIST 데이터 등을 사용가능하다.

```python
# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(), 
                          #transform은 이미지를 tensor에 맞게 조정하기 위하여 생성
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
```

- 패키지 설정

```python
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드설정
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
```

- MLP 구현

그다음 MLP를 구현하기 위해 필요한 개념들이다.

→ epoch : 모든 학습 데이터에 대해 순전파와 역전파를 진행한 것.

→ batch size : 순전파/역전파를 진행할 때 사용하는 학습 데이터 개수. 데이터 셋 내에 많은 양의 데이터가 있기 때문에, 그 데이터를 나눠 학습을 진행해야한다.

→ Learning rate : 학습률. 학습률이 클 수록 학습의 폭이 커진다.

→ optimizer : 손실함수의 최솟값을 찾는 것을 학습 목표로 하는 최적화 알고리즘을 옵티마이저라고 한다.

![한국딥러닝_순전파_역전파_MLP_구조.png](%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A9%E1%86%BC%E1%84%8C%E1%85%B5%E1%84%82%E1%85%B3%E1%86%BC%201-2%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%2095941e0cd9454495838d40f289b5aa74/%25ED%2595%259C%25EA%25B5%25AD%25EB%2594%25A5%25EB%259F%25AC%25EB%258B%259D_%25EC%2588%259C%25EC%25A0%2584%25ED%258C%258C_%25EC%2597%25AD%25EC%25A0%2584%25ED%258C%258C_MLP_%25EA%25B5%25AC%25EC%25A1%25B0.png)

→ 순전파 : 입력층부터 출력층까지 순서대로 변수들을 계산하고 저장하는 것.

![한국딥러닝_순전파_역전파_구조_4.png](%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A9%E1%86%BC%E1%84%8C%E1%85%B5%E1%84%82%E1%85%B3%E1%86%BC%201-2%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%2095941e0cd9454495838d40f289b5aa74/%25ED%2595%259C%25EA%25B5%25AD%25EB%2594%25A5%25EB%259F%25AC%25EB%258B%259D_%25EC%2588%259C%25EC%25A0%2584%25ED%258C%258C_%25EC%2597%25AD%25EC%25A0%2584%25ED%258C%258C_%25EA%25B5%25AC%25EC%25A1%25B0_4.png)

→ 역전파 : 순전파 과정을 역행함으로써, 오차를 기반으로 가중치 값들을 업데이트하기 위한 과정.