# 인공지능 2주차 - 2

추가 일시: 2024년 7월 8일 오후 3:29
강의: 인공지능

# 과제

주피터에 코드 누적(구현)

## OX_Classification

- MLP로 분류 진행하기
    - OX_Dataset 불러오기
    - OX_image 전처리
    - MLP로 진행시 k-fold적용 , 미적용 각각 정확도 산출

# acitvation function

## acitvation function란?

입력신호의 총합을 출력 신호로 변환하는 함수, 입력 신호의 총합이 활성화 시킬지 말지를 정함, 비선형 변환을 통해 신경망이 다양한 종류의 복잡한 함수를 학습한다.
→관련성 없는 데이터 포인트를 억제하고 중요한 정보를 사용하도록 돕는다

### 등장 배경

인공 신경망은 뉴런을 모방하였는데 뉴런은 일정세기 이상의 자극에만 신호를 전달하는 계단 방식을 사용했다.
→최초의 인공 신경망인 퍼셉트론은 계단 함수를 사용
→인관과 다르게 인공 신경망은 학습의 연속성을 표현하지 못해 뉴런과 다른길을 사용하기로 했다
→연속함수를 이용하여 학습이 연속적으로 학습하도록 했다
→따라서 값을 받고 다음 층으로 값을 전달하는 연속성이 있는 비선형 함수를 사용

### 비선형인 이유

선형함수를 사용하면 신경망의 층의 깊게하는 의미가 없어진다. 층이 깊어져도 결말은 선형이다
y(x)= h(h(h(x)))→ y(x)=c*c*c*x 은 y(x)=ax 와 같다
결국에는 은닉층이 필요없어지는 것이다

### 목적

입력값에 대한 출력값이 선형으로 나오지 않아서 선형 분류기를 비선형 시스템으로 만들 수 있다.

비선형이 필요한경우

![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/fe6f51ad-85b8-4421-a608-cfc47d51944a)

- 출처
    
    https://wikidocs.net/120076
    
    https://kevinitcoding.tistory.com/entry/활성화-함수-정의와-종류-비선형-함수를-사용해야-하는-이유
    
    https://ganghee-lee.tistory.com/30
    
    https://syj9700.tistory.com/37
    
    https://www.v7labs.com/blog/neural-networks-activation-functions
    

## 종류, 특징

### sigmoid

수많은 값들을 0과 1사이의 확률값으로 변환시킴
![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/789b12fa-f461-4d26-9ec3-0785d2de173f)

vanishing gradient문제: 입력값이 커질수록 기울기가 0으로 수렴하는 단점이 있다, 층이 깊어지면서 0에 가까운 기울기를 곱하는 형태가 되어 기울기가 소실되는 문제가 발생위험

함수 중심값이 0이 아닌 문제: 함수의 중심값은 0.5이다 때문에 모수 추정이 어려운 단점이 있다.

결과가 0,1인 이진 분류의 출력 레이어에서 사용한다. 결과가 0과 1사이에만 있어서 0.5보다 크면 1 작으면 0이 되는걸 쉽게 예측이 가능하다

### thah

쌍곡선 함수, 기울기가 0으로 수렴하는 구간 존재해 소실 문제가 존재
양,음수 모두 학습 효율성이 높다→출력값에 대한 부정, 중립, 긍정으로 매핑이 가능하다
시그모이드 함수대비 기울기가 작아지지 않는 구간이 넓다
→심층신경망에서 시그모드 함수 대안으로 활용

![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/ce4455f0-7587-416d-bd5f-e38d3fdbb845)

은닉층의 평균이 0에 가까워 데이터를 중심화 하는데 도움이 된다

### relu

입력값이 음수면 0 양수값이면 그대로를 보내는 함수, 0을 기준으로 활성/비활성이 이루어진
위 함수들의 기울기 소실 문제를 해결해준 함수이다.

제한된 뉴런만 활성화 되어→연산이 간편하고 층들을 깊게 쌓을 수 있는 장점이 있다
입력값이 너무 커지면 입력값에 편향이 될수 있어서 최대값을 6이하로 제한하는 방식을 사용하기도 함

![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/49e0fad6-de79-45db-b7b5-b7f31073cf19)

dying ReLU: 입력이 모두 음수이면 모두 0으로 만들어 다수의 뉴련이 비활성화가 될 수 있다.
이를 해결하기 위해서 dying ReLU가 형성되었다
x값이 0미만이어도 기울기가 0이 아니어 문제에서 벗어날 수 있다

![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/f2ab521a-8464-46b1-8afa-106b462f2a2f)

ReLu는 수학적 연산을 포함하기 때문에 위의 다른 함수들보다 계산 비용이 저렴하다, 따라서 다른 함수에 비해 훨씬 빠르게 학습이 가능하다

### softmax

시그모이드 함수와 비슷하게 0-1사이의 수로 변환하여 출력하고 출력값들의 합이 1이 되는 함수이다→가장큰 확률에 1을 반환하고 나머지는 0을 반환한다
→다중 분류의 최송 활성화 함수로 사용된다
결과값들이 독립적인 시그모이드와 비슷하지만 확률값들이 독립적인 시그모이드와 다르게 연관되어있어 다중분류문제에서 활용이 가능하다


![Untitled 5](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/0e7d5a21-e157-4124-b09f-d10c3a4ffb2f)

- 출처
    
    https://kevinitcoding.tistory.com/entry/활성화-함수-정의와-종류-비선형-함수를-사용해야-하는-이유
    
    https://velog.io/@bandi12/활성화-함수의-역할과-종류
    
    https://syj9700.tistory.com/37
    
    https://velog.io/@ym980118/딥러닝-활성화-함수-간단-정리-시그모이드-소프트맥스-ReLU
    
    https://wikidocs.net/35476
    
    https://www.geeksforgeeks.org/activation-functions-neural-networks/
    
    https://www.v7labs.com/blog/neural-networks-activation-functions
    
    https://towardsdatascience.com/what-is-activation-function-1464a629cdca
    

# k-fold

## k-fold란?

훈련 데이터셋을 k개의 폴드로 나눈뒤 k-1개의 폴드를 훈련 셋으로 나머지 1개를 테스트 셋으로 성능을 평가하는 방식이다
이방식을 k번 반복해서 k개의 모델과 성능을 얻는다

### 단점

일반적인 학습법에 비해 시간이 오래걸린다

![Untitled 6](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/06941bdf-7564-4a9a-ae26-d94f19d5e586)

### 사용이유

데이터의 모든 부분을 테스트 데이터의 일부로 사용이 가능하다. 따라서 작은 데이터 세트의 모든 데이터를 훈련과 테스트에 사용이 가능해 모델의 성능으 더 잘 평가가 가능하다
과적합을 피하는데 도움이 된다

- 출처
    
    https://blog.naver.com/sjy5448/222427780700
    
    https://jellyho.com/blog/84/
    
    https://nonmeyet.tistory.com/entry/KFold-Cross-Validation교차검증-정의-및-설명
    
    https://www.kdnuggets.com/2022/07/kfold-cross-validation.html
    
    https://www.analyticsvidhya.com/blog/2022/02/k-fold-cross-validation-technique-and-its-essentials/
    

## k-fold 원리

1. k개 만큼의 같은 크기의 데이터 집합으로 나눈다
2. 특정 집합은 데이터 평가용으로 사용하고 나머지는 학습용으로 사용한다
3. 이 방식을 k번 반복하는데 평가용 데이터 집합은 변경하면서 진행한다.
- 출처
    
    https://incodom.kr/k-겹_교차_검증