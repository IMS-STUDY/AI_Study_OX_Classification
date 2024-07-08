# 인공지능 7/4

비용함수 : 오차를 줄이기 위함

손실함수

과적합 방지 : 정규화를 자주 씀

- [x]  완전 연결 계층의 정의, 문제점
- [x]  합성곱 정의, 문제점?
- [x]  패딩 정의, 종류
- [x]  풀링 사용이유, 특징, 장단점
- [x]  conv(n)d [n=1,2]
- [x]  입력데이터 차원, kernel
- [x]  파이토치에서 MLP 구현하는 법

# 완전 연결 계층

- 정의
    
    ![img](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/6d838b5b-dd24-42ba-beea-f4564a5eb6e4)
    
    한 층의 모든 노드가 다음 층의 모든 노드와 연결된 상태를 말함.
    
    Dense layer라고도 함.
    
    입력과 출력 사이의 매핑을 생성하며, 각 연결은 가중치를 가지며, 노드는 bias를 가질 수 있음. 이러한 Dense Layer는 신경망이 복잡한 패턴을 학습하는 데 있어 중요한 역할을 함.
    
    1차원 배열의 형태로 평탄화된 행렬을 통해 이미지를 분류하는데 사용되는 계층
    
    2차원 배열 형태의 이미지를 1차원 배열로 평탄화
    
    활성화 함수 뉴런을 활성화
    
    분류기(softmax)함수로 분류
    
- 문제점
    - 데이터의 형상이 무시됨
    - 입력 데이터가 3차원이어도 평평한 1차원 데이터로 평탄화해줘야 함
    - 완전 연결 계층은 형상을 무시하고 모든 입력 데이터를 동등한 뉴런(같은 차원의 뉴런)으로 취급하여 형상에 담긴 정보를 살릴 수 없음
    
- 참고자료
    
    [완전 연결 계층 - MATLAB
    - MathWorks 한국](https://kr.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer_ko_KR.html)
    
    [완전 연결 계층, Fully connected layer](https://dsbook.tistory.com/59)
    

# 합성곱

- 정의
    
    ![img 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/fb38e072-5962-48b1-ba7f-7eb1a1620da5)
    
    시간의 흐름에 따라 g(x)가 이동하면서 f(x)를 평균적으로 얼마나 변화시키는지 나타내는 것으로 볼 수 있음.
    
    입력된 이미지의 특징을 찾기 위해 필터를 씌우는 과정
    
    CNN 신경망의 핵심으로 이미지 처리에 사용되는 연산이다.
    
- 문제점
    
    합성곱을 진행할수록 크기가 점점 작아지고, 가장자리에 있는 픽셀의 정보가 점점 사라지는 현상이 발생하며, 합성곱을 더 이상 적용할 수 없게 됨.
    

# Padding

- 정의
    
    입력 데이터의 주변을 0으로 둘러싸 데이터 손실을 방지하는 것
    
- 종류
    - 밸리드 패딩 : 패딩을 추가하지 않음
    
    | 1 | 2 | 3 |
    | --- | --- | --- |
    | 4 | 5 | 6 |
    | 7 | 8 | 9 |
    - 풀패딩 : 입력데이터가 mxm, 필터가 nxn일 때 n-1픽셀의 패딩을 추가
    
    | 0 | 0 | 0 | 0 | 0 |
    | --- | --- | --- | --- | --- |
    | 0 | 1 | 2 | 3 | 0 |
    | 0 | 4 | 5 | 6 | 0 |
    | 0 | 7 | 8 | 9 | 0 |
    | 0 | 0 | 0 | 0 | 0 |
    
    | 1 | 0 |
    | --- | --- |
    | 0 | 1 |
    
    필터
    
    m = 3, n = 2이므로 1픽셀의 패딩 추가
    
    - 세임패딩 : 위의 내용에서 (n-1)/2픽셀 만큼의 패딩을 추가
    
    | 0 | 0 | 0 | 0 | 0 | 0 |
    | --- | --- | --- | --- | --- | --- |
    | 0 | 1 | 2 | 3 | 4 | 0 |
    | 0 | 5 | 6 | 7 | 8 | 0 |
    | 0 | 9 | 10 | 11 | 12 | 0 |
    | 0 | 13 | 14 | 15 | 16 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 |
    
    | 1 | 0 | 1 |
    | --- | --- | --- |
    | 0 | 1 | 0 |
    | 1 | 0 | 1 |
    
    필터
    
    m = 4, n=3이므로
    
    (3-1)/2 = 1픽셀의 패딩 추가
    

# Pooling

convolution층의 출력 데이터의 크기를 줄이거나 특정 데이터를 강조함.

- 사용하는 이유
    
    피처맵의 크기를 줄이고 차원을 축소함. 이를 통해 연산량 감소 가능.
    
    이를 통해 과적합을 제어할 수 있음.
    
- 특징
    - 필터의 사이즈 f, 스트라이드의 크기 s 두 개의 하이퍼파라미터를 가짐
    - 2x2를 자주 사용하며 이를 통해 들어오는 이미지의 높이와 너비를 절반만큼 줄임
    - 이미지의 채널과 풀링을 통해 나온 결과의 채널 수는 같음
    - input feature map에 변화(shift)가 있어도 pooling의 결과는 변화가 적다. (robustness)
    - 패딩은 거의 사용하지 않음
    - 학습할 수 있는 매개변수가 존재하지 않
    - Max Pooling : 각 영역의 최댓값
    - Average Pooling : 각 영역의 평균값
- 단점
    - 풀링 과정에서 정보의 손실이 발생할 수 있음
    - 입력 데이터의 공간적인 정보를 정확히 보존하지 못할 수 있음
    

[Pooling을 사용하는 이유, pooling의 특징, pooling의 효과 (CNN, Sub sampling, Max pooling, Average pooling)](https://technical-support.tistory.com/65)

[Stride와 Pooling의 비교](https://gaussian37.github.io/dl-concept-stride_vs_pooling/)

# conv(n)d [n = 1, 2, …]

CNN의 핵심 layer, 일반적으로 이미지에 존재하는 패턴을 인식하는 데 사용되는 딥 신경망의 한 종류

입력데이터의 특징을 추출

합성곱연산 일어

이미지 데이터의 공간적 상관관계 추출

많이 쌓일수록 더 넓은 영역의 상관관계를 추출함 동시에 데이터를 더 많이 압축하게 됨

가장 일반적으로 사용되는 합성곱 유형은 2D 합성곱 계층임

conv2D라고 함

합성곱 연산을 진행하고 2D 피처 행렬을 다른 2D 피처 행렬로 변환함.
![trans_conv](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/65e5bc44-fab9-4eb2-9be1-6d2f0d0b2e98)

# 입력데이터 차원, kernel

차원=변수의 수로 볼 수 있음

![img1 daumcdn](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/66ea469a-f0de-463c-8067-32237e204094)

변수의 수가 늘어나면 차원이 커짐

이로인해 차원의 저주 발생.

변수의 수가 많아짐→차원 커짐→데이터 분석을 위한 필요 데이터 건수 많아짐

Kernel은 두 벡터의 내적(inner product)이며, 기하학적으로 cosine 유사도를 의미하기 때문에 Similarity function 이라고도 불림

# Pytorch에서 MLP 구현하는 법

1. 필요한 패키지들을 불러옴
2. 데이터를 불러옴
3. 데이터 전처리, 데이터를 트레이닝, 테스트로 분리
4. 파라미터를 설정
    - epoch : training example에 대한 forward pass, backward pass를 진행
    - batch size : 한 번 forward pass / backward pass할 때 사용하는 training example의 개수
    - Iteration : batch size를 기준으로 한 epoch을 진행할 때 몇 번을 반복해야 하는지에 대한 값
5. 모델 구축, 모델, 손실함수, 최적화 방법 설정
6. 학습 진행

- 참고자료
    
    [Pytorch로 구현하는 Multi-Layer Perceptron](https://data-science-hi.tistory.com/188)