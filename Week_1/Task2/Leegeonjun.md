# 인공지능 1주차 - 2

추가 일시: 2024년 7월 4일 오후 4:18
강의: 인공지능

# 과제

데이터셋 O-X 인당 20개씩 그리기 ( O 20개, X 20개 , 300x300)

# CNN이론

## 완전 연결 계층

### 완전 연결 계층 이란?

CNN의 계층중 하나, 앞에서 한 작업들의 분류를 결정하는 단계이다

완전연결되었다: 한층의 모든 뉴런이 다음층의 모든 뉴런과 연결된 상태
2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층이다.
최종 FC 계층의 뉴런수는 일반적으로 분류 문제의 출력 클래스 수와 일치 한다, 10개의 클래스 숫자 분류 문제의 경우 최종 FC계층에는 10개의 뉴런이 있고 각각은 클래스중 하나에 대한 점수를 출력한다.

FC의 과정

1. flatten: 각 레이어를 1차원 벡터로 전환(이미지→1차원)
2. FC: 1차원으로 변환된 레이어를 하나의 벡터로 연결
3. Softmax함수를 이용해 가장 높은 class를 output으로 분류

softmax함수
k개의 벡터를 입력하면 각 값들은 정규화 하여 0-1사이의 값으로 나타나고 각 값들의 합은 1이다. 
여기서 지수함수를 넣어서 정규화를 하기때문에 큰값은 극도로 커지게 되어 구분이 쉬워지고 가장 큰값이 있는것이 해당이미지의 class이다.

![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/c9b1cf7c-2d5b-442b-b088-6c41df5c2004)

### 문제점

흑백이미지를 다루는 경우 색 데이터를 하나만 다루기 때문에 벡터를 1차원 행렬로 변환시키는데 어려움이 없다
하지만 RGB필터의 픽셀값을 가진 컬러 이미지라면 3차원의 이미지를 1차원으로 평탄화 해야하기 때문에 공간 정보가 손실이 되어 손실로 인한 정보부족으로 인해 이미지를 분류하는데 한계가 있다.

## 합성곱

### 합성곱이란?

FC와 다르게 3차원 데이터를 받고 변경하는 것이 아닌 다음 계층에 3차원 데이터로 전달한다.
합성곱은 입력 데이터에 필터를 적용한다. 
합성곱에서 고려할 3가지
1. 필터의 크기
2. 패딩
3. 스트라이드
풀링은 합성곱 계층 다음의 계층이다

![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/d1da8306-3a5d-41fa-ac6d-3ba49f579e52)

![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/b643c59e-460f-4fa5-b74e-80fd314dcbed)

### 필터의 사용

필터를 이용하여 입력 이미지에서 모서리, 질감, 요양 및 복잡한 패턴을 감지할 수 있다. 계층적으로 들어가며 복잡성이 증가하면 입력 이미지의 복잡한 부분을 학습이 가능하다.
사진의 예시
처음에는 필터를 통해 수평선, 수직선 같은 저수준의 특징을 학습하고 층이 깊어질수록 더 디테일한 특징을 학습한다.

![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/fa7dbe4c-3e52-48f1-8f4e-3d436226d762)

버스 사진이 필터를 통과하고 활성화 된부분을 보여준다
(빨간: 수평적 모서리, 파랑: 수직적 모서리)

![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/a69767c4-ca1a-401f-9bb3-e3a5516d1717)

어느정도 층이 쌓인상태이다
(빨간색: 머리카락, 녹색: 얼굴관련1, 파란색: 얼굴관련2)

## padding

### padding 이란?

출력 크기를 조정하기 위해 합성곱 이전에 데이터 주변 값을 0으로 체운다
(4,4) 데이터에 (3,3) 필터를 이용하면 (2,2)인 결과가 나온다
이렇게 크기가 점점작아지면 출력크기가 1이되고 이이상은 합성곱 연산을 못하는 경우가 생긴다→패딩을 이용해 입력 데이터의 크기를 고정한상태로 다음 계층에 전달이 가능하다

![Untitled 5](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/3d767238-a647-4546-988b-1a45cf6b7003)

### padding 종류

밸리드 패딩

패딩을 추가하지 않은 형태, 밸리드 패딩을 적용한것이다.
→입력보다 결과가 작아짐


![Untitled 6](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/c446cef4-7778-432b-839c-58f75195ecd0)
입력데이터의 요소들이 같은 비율로 사용되지 않는 문제존재
→모서리 부분의 데이터가 조금만 반영

풀 패딩

입력데이터의 모든 원소가 합성곱 연산에 같은 비율로 적용되게 패딩의 적용

![Untitled 7](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/d7e780f3-a5a7-43b7-9849-b0dac8d9e595)

세임 패딩

입력크기와 출력 크기가 동일하게 유지하는 패딩

![Untitled 8](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/3dff26ca-7f82-44cf-bf4e-5343ddc3c435)

## pooling

### 사용이유

다음 레이어에 들어가는 이미지의 크기를 크게 감소시킨다
→학습시간이 단축

![Untitled 9](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/8dc74623-0b85-41eb-ae92-c11d2f459290)

메모리 효율성: 가장 일반적으로 사용하는 최대 풀링과 평균 풀링은 피처 맵의 크기 감소에 도움이 된다. 가장 중요한 정보는 유지하면서 크기를 감소하여 메모리 효율성에도 좋고 계산적으로도 더 빠르게 만든다.
변환 불변성: 특징의 존재의 집중하도록 하여 변환 불변성에 기여한다. → 특정 특징을 향상이 가능(최대값 사용시)

### 특징 및 장단점

특징

1. 차원축소: 매개변수를 줄이고 계산 복잡도를 줄임
2. 변환(이동) 불변성: 특정위치의 변화에 강해짐
3. 특성계층: 풀링 계층은 하위수준 특성을 결합하여 상위 수준의 특성을 형성
4. 정규화: 과적합을 줄여 정규화의 한 형태로 사용가능

장점

1. 계산량과 메모리 사용량 감소
2. 과도한 적합을 방지: 가중치를 증가시키지 않아 과도한 적합을 방지한다
3. 특징 추출: 노이즈 제거 및 압축하여 유용한 특징을 추출

단점

1. 정보손실: 데이터가 감소하여 모호해질 위험
2. 정확도감소
3. 중앙속성저하: 적용 전에는 각 픽셀이 특징에 대응하지만 풀링에는 픽셀의 특징을 나타내는 출력 값이 있기 때문에 풀링 계층에서 서로 다른 픽셀 값을 입력하면 동일한 출력 값으로 압축되어 중앙속성이 저하되는 문제가 발생 

## conv(n)d [ n = 1, 2 ]

### 입력데이터의 차원, kernel

**커널**

커널은 컨볼루션시 이미지에서 특징을 추출할때 사용하는 도구이다(아래 사진에서 커널을 이용하여 이미지가 더 선명해진다)
+노이즈가 들어가있는 것을 깔끔하게 만들어준다

![Untitled 10](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/538cad69-87d1-43af-a02b-ada2d1b7f8b1)

**1차원컨볼루션**

시간적 데이터같은 1차원 데이터를 처리하는데 적합하여 텍스트, 오디오같은 분야에 자주 사용한다. 1차원 입력데이터는 한가지 축에 따라 배열되어 있고 커널도 같은 차원에서 데이터를 슬라이드하며 합성곱 연산을 적용한다.


![Untitled 11](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/0ede4387-aa50-4fae-9ae8-9649233e699f)

1차원 컨볼루션을 사용하는 이유
오디오같은 시간에 따라 변화는 데이터 분석에는 시간축으로 데이터가 형태되어 있는 1차원 컨볼루션의 데이터의 패턴을 효과적으로 학습이 가능하다

**2차원컨볼루션**

가로와 세로가 있는 2차원의 격자구조의 데이터를 처리하는데 좋다. 가로 세로 두방향을 고려하면 필터는 슬라이딩을 한다.


![Untitled 12](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/e685e190-d9f1-42e5-bdea-648d2afa54bc)

# pytorch에서 MLP 구현하는법(이론)

## 용어

### 퍼셉트론

인간의 뉴련과 비슷하게 작동하도록 고안된 뉴런이 하나뿐인 간단한 형태의 신경망이다. 함수들을 이용하여 입력세기를 인식하고 임계값을 초과시 출력신호를 보내도록 함

### 활성화 함수

뇌에서 결정을 내리는 역할, 선형결합한 가중값이 정해진 임계값보다 크면 뉴련을 활성화 한다

## 학습

### 퍼셉트론 학습

1. 뉴런이 입력을 선형결합하여 가중값의 합을 구하고 활성화 함수에 입력해 예측값을 구한다
2. 예측값과 실제 값을 비교해 오차를 구한다
3. 오차에 따라 가중치를 조정한다
4. 1-3과정을 반복해 오차가 0에 가까워지도록 조정한다

## MLP구현하기

1. 필요한 패키지 import, 필요한 device할당
2. 데이터셋으로 DataLoader 생성하기
3. MLP모델 생성
픽셀을 부동 소수점 값으로 변환→이미지 픽셀 값을 255로 나눈다.(최대 256이므로 0을 제회한 255로 나누면 0-1값으로 된다)
데이터 시각화
4. 평가함수 생성
5. 학습 진행
6. 테스트 데이터로 모델 성능 확인

# 출처

https://blog.naver.com/PostView.nhn?blogId=intelliz&logNo=221709190464

https://m.blog.naver.com/bananacco/221928562116

https://velog.io/@grovy52/Fully-Connected-Layer-FCL-완전-연결-계층

https://medium.com/@vaibhav1403/fully-connected-layer-f13275337c7c

https://dsbook.tistory.com/59

https://velog.io/@bbirong/밑딥-7장.-합성곱-신경망CNN

https://www.linkedin.com/pulse/why-we-prefer-convolution-neural-networks-cnn-image-data-salunke-a1czc

https://datascience.stackexchange.com/questions/15903/why-do-convolutional-neural-networks-work

https://betterprogramming.pub/why-you-should-use-convolutions-in-your-next-neural-net-using-tensorflow-37d347544454

https://ardino.tistory.com/40

https://yifan-online.com/en/km/article/detail/11836

https://www.dremio.com/wiki/pooling-layers/

https://mz-moonzoo.tistory.com/64

https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37

https://velog.io/@7ryean/CNN-Convolution-연산

https://velog.io/@hipjaengyi_cat18/Perceptron을-이해하고-Pytorch로-MLP-구현해보기-내가보려고정리한AI

https://www.turing.com/kb/multilayer-perceptron-in-tensorflow