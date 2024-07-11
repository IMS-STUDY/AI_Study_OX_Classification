# 3 - Activation Function, k-fold

# Activation Function

### 개요

> **활성화 함수**(活性化函數, [영어](https://ko.wikipedia.org/wiki/%EC%98%81%EC%96%B4): activation function)는 [인공 신경망](https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5_%EC%8B%A0%EA%B2%BD%EB%A7%9D)에서 입력을 변환하는 [함수](https://ko.wikipedia.org/wiki/%ED%95%A8%EC%88%98)이다.
출처 - https://ko.wikipedia.org/wiki/활성화_함수
> 

입력 신호의 총합을 출력 신호로 변환하는 함수

대표적으로 Sigmoid, ReLU, Tanh, Softmax 함수 등이 있음

### 목적

비선형 함수를 이용하여 여러 층으로 구성된 은닉층들 사이에 배치함으로써 비선형 문제를 해결하기 위함

비선형 문제? → 직선으로 표현할 수 없는 문제

### 종류

1) Sigmoid

![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/3d5575b5-7db8-45d3-b371-2a9f27e7b23a)

- 선형함수의 결과를 0부터 1까지 비선형 형태로 변환하는 함수
- 모델이 복잡해질수록(레이어가 많을수록) 기울기가 소실되는 Gradient Vanishing 현상 발생
    - Gradient Vanishing 사진
        
        ![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/19807cd0-108f-4cf9-b9b6-717e1a9ba285)
        

2) Tanh

![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/1531e2c1-8a8e-4171-8779-b4987180095a)

- Sigmoid와 유사하지만 범위가 -1 ~ 1로 변경됨
- Sigmoid의 상위호환 격이지만 여전히 기울기 소실 문제 발생

3) ReLU

![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/b4a09372-6ddd-4b9c-85bc-86056d6ea9ae)

- 입력이 양수일때는 입력값 그대로, 음수일때는 전부 0을 출력하는 함수
- 매우 빠른 학습 속도
- Gradient Vanishing 문제 해결
- 단, 음수 값을 무조건 0으로 처리하므로 음수 값에 대한 약점 존재
    
    → 이 단점을 보완한 Leaky ReLU 함수가 있다.
    

4) Softmax

![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/61f04e92-b1b4-4a4b-86cd-8f27a012bae7)

- 입력받은 값을 항상 0 ~ 1 사이로 정규화시키며 총합을 1로 고정시키는 함수
- 주로 출력 노드(Output Layer)의 활성화 함수로 쓰임

# K-Fold Cross Validation

### 개요

![Untitled 5](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/a60e48fc-0d29-462a-90e7-eb4ed0a4633e)

- 훈련 과정에서 훈련 세트를 K개의 Fold로 나누어서 교차 검증하는 방식
- 하이퍼파라미터를 효율적으로 설정할 수 있는 장점

### 원리

![Untitled 6](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/a9fe3315-9354-4ce2-9b0c-21b8d8009c76)

과정)

1. 훈련 세트를 K 개의 Fold로 나눈다.
2. K개 만큼의 Split을 만들고, n번째 Split의 n번째 Fold를 Validation Fold라고 설정한다.
3. 각 Split마다 훈련을 진행한다.
4. 훈련 결과를 종합하여(평균 등) 최적의 파라미터를 선정한다.