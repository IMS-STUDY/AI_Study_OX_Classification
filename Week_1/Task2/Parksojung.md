# 7/4

*[과제]*

*1주 -2*

- *데이터셋 O-X 인당 20개씩 그리기 ( O 20개, X 20개 , 300x300)*

# *CNN 이론*

Convolution Neural Network(합성곱 신경망)

→ Deep Neural Network에서 이미지나 영상과 같은 데이터를 처리할 때 발생하는 문제점들을 보완한 방법

*[번외] 왜 MLP가 아니라 CNN?*

![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/5994aae8-ac7c-4b68-ad78-44b49931c652)

위 사진을 사람은 같은 X라고 판단하지만 MLP를 사용하면 MLP에서는 이미지 픽셀 1개가 1개의 노드 이므로 이미지가 shift라게 되면 노드의 값들이 많이 바뀌어 가중치의 값들이 무력화 됨.

따라서 CNN은 feature extraction 과정을 통해 이미지를 학습하게 됨.

![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/d3e36bc3-1f8c-45b0-b7e2-a44561f165e3)

CNN은 필터값을 딥러닝 network 구성을 통해 이미지 분류 등의 목적에 부합하는 최적의 필터 값을 학습을 통해 스스로 생성함.

![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/0d9e16d1-f43a-43d1-bbfe-7f6f73d530da)

## *완전 연결 계층*

1. ***완전 연결 계층이란?***
- Fully connected layer. 한 층(layer)의 모든 뉴런이 그 다음 층(layer)의 모든 뉴런과 연결된 상태를 말함.
- 1차원 배열의 형태로 평탄화된 행렬(3차원을 1차원으로 flatten)을 통해 이미지를 분류하는데 사용되는 계층

```python
model = keras.Sequential()
model.add(layers.Flatten(input_shape = (28,28)))
model.add(layers.Dense(128, activation='relu'))
model.asdd(layers.dense(10, activation='softmax'))
```

위 코드로 평탄화를 통해 흑백 이미지, 2차원 벡터의 행렬을 1차원 배열로 평탄화하고, ReLU함수로 뉴런을 활성화하여 softmax함수로 이미지를 분류하는 것 까지가 완전 연결 계층이라고 할 수 있음.

1. ***완전 연결 계층의 문제점***
- 이미지 데이터의 형상을 유지하지 못함

Fashion-MNIST(의류 분류 데이터 셋으로 10개의 카테고리로 분류)에서 사용한 이미지는 흑백이미지라 명암으로만 이미지가 구성되어 RGB를 사용하는 컬러 이미지와 달리 벡터를 1차원 행렬로 변환 시키는데 어려움이 없음.

![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/6b645b6f-c5d1-499e-9d1b-7349e65ce6ce)

fashion-mnist (60000개의 training-set 가짐)

(60000, 28, 28)에서 60000은 이미지의 갯수, 28, 28은 가로 세로 크기를 나타냄.

이미지가 RGB 필터의 픽셀값을 가진 컬러 이미지였다면 (6000, 3, 28, 28)의 4차원 넘파이 배열로 나타났을 것. (여기서 3은 RGB 채널 개수)

이럴 경우 3차원 이미지를 1차원으로 평탄화하면 공간 정보가 손실 될 수밖에 없으며, 정보 부족으로 인해 이미지를 분류하는데 한계가 생길 수 밖에 없음!

## *합성곱*

1. ***합성곱이란?***

Convoluiton. 함수를 서로 곱해서 합함.

![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/66e44b40-23a3-41ba-8879-0c3cee199b56)

정의 식

어느 한 영역의 상태를 바꾸고 싶을 때, 주변 영역들에 일정한 가중치를 두어 전부 섞어 그 결과로 상태를 바꾸는 것.
“이미지에 필터를 씌운다” 라고 생각하면 편함.

[과정]

1) 이미지를 준비함. 이미지는 0부터 255까지 존재하고 255에 가까울 수록 밝음.

![Untitled 5](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/12ca8871-333f-4809-abe2-3d486879d8b8)

2) 커널 (필터)를 준비함. 크기와 내부에 값은 사용자 마음임.

![Untitled 6](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/1716f65f-8b24-434c-885f-0ddc9b3bced5)

3) 커널과 이미지를 왼쪽 상단에 겹쳐서 서로 맞닫는 부분의 곱을 전부 구하고 그것들을 전부 합함.

![Untitled 7](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/76af207f-51b4-402c-a783-2c72c0f24e15)

연산하면 125

4) 이후 커널을 한칸 오른쪽으로 이동 시키고 (3)을 반복함. 만약 커널이 더 이상 이미지의 오른 쪽에 가지 못하면 맨 왼쪽으로 돌아가서 한칸 아래로 내려감. 내려가지도 못할 경우에는 지금까지 (3)에서 연산한 값을 저장한 배열을 반환.

![Untitled 8](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/57d3046c-6bca-4bee-b50d-c4ade0ed40bc)

1. ***합성곱 문제점***
- 반환된 배열의 테두리가 비어 있음(이미지 크기가 줄어듦)
→ padding 어떻게 처리할지 정해야 됨

## *padding*

1. ***padding 이란?***

합성곱을 수행하기 전에 주어진 이미지를 0으로 둘러싸는 작업.

padding을 함으로써 이미지의 가장자리가 손질되지 않고 원본을 유지할 수 있음.

![Untitled 9](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/23b7f632-1f4e-4d67-95fa-97195e23462b)

![Untitled 10](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/a8642f79-9c50-46b3-a1b0-09c7dd4506ac)

1. ***padding 종류***
- 밸리드 패딩(valid padding)
    - 패딩을 추가하지 않은 형태를 밸리드 패딩을 적용했다고 볼 수 있음
    - 밸리드 패딩을 적용하고 필터를 통과 시키면 결과는 입력 사이즈보다 작게 됨
    - 하지만 합성공 연산시 입력 데이터의 모든 원소가 같은 비율로 사용되지 않는 문제점이 있음 → 원소별 연산 참여 횟수를 결정하는 ‘패딩’과 ‘스트라이드’임
    
    *스트라이드: 필터를 움직이는 보폭, 대부분 1씩 이동하지만 조정을 해서 다른 크기만큼 이동 가능. 이에 따라 원소별 연산 참여 횟수가 달라지고 결과값도 다르게 나옴.
    
- 풀 패딩(full padding)
    - 입력 데이터의 모든 원소가 합성곱 연산에 같은 비율로 참여하도록 하는 패딩 방식

![Untitled 11](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/29f2773e-f648-4fcb-97a3-5854200b778e)

- 세임 패딩(same padding)
    - 출력 크기를 입력 크기와 동일하게 유지함
    - 입력 데이터 (h,w) 필터(f,f)가 있을 때 세임 패딩의 폭은 P=(f-1)/2 가 됨 → 절반 패딩(half padding)이라고도 부름

## *pooling*

2차원 데이터의 세로 및 가로 방향의 공간을 줄이는 연산

[종류]

- 최대 풀링(max pooling)
    - 대상 영역에서 최댓값을 취함
    - 이미지 인식 분야에서 주로 사용

![Untitled 12](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/4a678853-041d-4341-a491-2b00ca3838ca)

- 평균 풀링(average pooling)
    - 대상 영역의 평균을 계산
1. ***사용이유***
- pooling layer가 없으면 너무 많은 가중치(weight parameter)가 생겨 overfitting 가능성이 커짐. 따라서 데이터에서 특징만 추출하게 보완을 해서 overfitting을 방지하는 것임

1. ***특징 및 장단점***
- pooling layer에서 pooling은 대상 영역에서 최댓값 또는 평균을 취하는 것이므로 특별히 학습할 것이 없음
- pooling 연산은 입력 데이터의 채널 수 그대로 출력 데이터로 보냄.
pooling은 2차원 데이터의 크기를 줄이는 연산이라 3차원을 결정하는 채널 수는 건드리지 않음.
- pooling layer는 입력의 변화에 영향을 적게 받음. 입력 데이터가 조금 변해도 pooling layer 자체가 그 변화를 흡수하여 사라지도록 함.

## *conv(n)d [ n = 1, 2 ]*

n→ 합성곱을 진행할 입력 데이터의 차원. 합성곱 진행 방향을 고려해야한다는 소리.

1. **1차원 컨벌루션 - conv1d**
    - 이미지가 아닌 시계열 분석(time-seies analysis)나 텍스트 분석을 하는데 주로 많이 사용.
    - 여기서 1차원이란 합성곱을 위한 커널과 적용하는 데이터의 시퀀스(연속된 데이터)가 1차원의 모양을 가진다는 것을 의미
    - 합성곱 진행 방향이 한 방향(가로)
    - kernel size - 중앙값을 기준으로 양쪽에 대칭적으로 적용하기 때문에 일반적으로 홀수임.

![Untitled 13](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/9901c9cb-590b-4d0a-84b2-58f6b3ed7d5c)

conv1d

1. **2차원 컨벌루션 - conv2d**
    - 주로 이미지 처리에 사용됨
    - 합성곱 진행 방향이 두 방향(가로, 세로)
    - kernel size - convolution의 시야를 결정. 보통 3*3 픽셀로 사용.

![Untitled 14](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/863b28de-bc7b-4a81-9c3f-a566e211abaf)

conv2d

- *kernel function*
    - SVM(support vector machine)과 같은 머신러닝 모델에서 사용되는 함수
    - 적분값이 1이고, 원점을 중심으로 대칭인 non-negative(커널 함수의 값을 항상 0 또는 양수로 보장) 함수
    - 커널 함수는 두 개의 입력 벡터를 받아 두 벡터 간의 유사도 또는 내적 값을 계산하는 역할
    - 유사도 또는 내적 값으로 입력 데이터를 고차원 공간으로 매핑하거나 유사도를 측정하여 머신러닝 모델에서 판별 경계를 만들 때 사용

# *pytorch에서 MLP 구현하는법 (이론)*

https://velog.io/@hipjaengyi_cat18/Perceptron을-이해하고-Pytorch로-MLP-구현해보기-내가보려고정리한AI

1) 필요한 패키지 import 및 device 할당

- pytorch버전 확인, device에 GPU, M1 GPU, CPU 등을 할당

2) MNIST 데이터 셋으로 DataLoader(배치 학습에 사용) 생성

- torch.utils.data.DataLoader() 사용
- 또는 DataLoader import 하고 DataLoaer() 사용

3) MLP 모델 생성

- 클래스 생성하고 torch.nn.Module 상속 받음
- **__init**__ 에서 layer 결정
- forward에서 데이터가 어떻게 전달되는지 결정

4) 평가해줄 함수 생성

5) 학습하기

6) test 데이터로 모델 성능 확인하기