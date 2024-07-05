# 인공지능 7/1

- 딥러닝 개요
    
    딥러닝은 인공지능, 그 안에서도 머신러인의 한 분야.
    
    이미지 분류
    

# MLP(Multi Layer Perceptron)

![1_MF1q2Q3fbpYlXX8fZUiwpA](https://github.com/IMS-STUDY/AI-Study/assets/127017020/88bd69cd-31ad-4daa-83ab-f1646b1be1bd)

- 개요
    
    단층 퍼셉트론은 XOR문제를 해결할 수 없음.
    
    이를 해결하기 위해 MLP가 도입됨
    

- MLP(Multi - layer perceptron)
    
    ![nodeNeural](https://github.com/IMS-STUDY/AI-Study/assets/127017020/bc776428-925c-4673-9b4c-4409719afe9c)
    
    지도학습에 사용되는 인공 신경망의 한 형태
    
    MLP는 입력층, 은닉층, 출력층으로 구성됨.
    
    은닉층이 하나 이상이면 MLP
    
    층과 층 사이의 연결은 있지만 같은 층 안에서의 연결은 없음.
    
    입력된 데이터에서 복잡한 패턴을 추출하는데 도움이 됨
    
    은닉층의 노드가 많아지면 정확도가 올라가지만 학습 속도도 올라가고 과적합이 발생할 수 있음
    
    - 과적합 : 모델의 성능을 하락락시키는 주요 원인. 훈련 데이터를 과하게 학습해서 훈련 데이터에 대한 정확도는 매우 높지만 새로운 데이터, 테스트 데이터에 대해서는 제대로 동작지 않음. 복잡한 모델에서 과적합 발생 확률이 더 높음.
        
        해결 방법으로는 데이터의 양을 늘리거나 모델의 복잡도를 줄이고 Regularization을 적용하거나 Dropout을 이용하는 방법이 있음.
        
        **데이터 양을 늘림**→데이터의 양이 적으면 데이터의 패턴이나 노이즈까지 학습하게 되는데 이를 예방할 수 있음.
        
        **모델의 복잡도를 줄임**→모델의 복잡도가 너무 낮거나 높으면 과적합 문제가 일어남. 복잡도는 모델의 파라미터 수. 따라서 적당한 복잡도의 모델을 사용해야 함.
        
        **Regularization(일반화)**
        
        - L1규제 : 가중치들의 절대값 합계를 비용함수에 추가.
        - L2규제 : 모든 가중치들의 제곱합을 비용함수에 추가
            
            -비용함수 : cost를 줄이는 함수
            
            -손실함수 : 모델의 성능을 측정하는 데 사용되는 함수
            
        
        **드롭아웃**→학습 과정에서 일부 노드를 사용하지 않는 방법. 
        
    
    주로 분류, 회귀 문제에 적용되며, 학습 알고리즘으로 역전파가 주로 사용됨.
    
    - backpropagation(역전파) : 역전파는 뉴런 간 연결의 가중치를 조정하여 네트워크를 훈련하는 데 사용되는 지도 학습 알고리즘. 출력층에서 입력층 방향으로 진행 하며 가중치를 업데이트.
        
        연쇄법칙으로 해결.
        
    
    입력값과 가중치에 대하여 각 층에서 활성화 함수를 적용해 결과를 도출함.
    
    단층 퍼셉트론 개선 → 비선형 활성화 함수 사용. (시그모이드,LERU 등)
    
    - 활성화 함수 : [인공 신경망](https://en.wikipedia.org/wiki/Artificial_neural_network) 에서 노드의 활성화 **함수는** 노드의 개별 입력과 가중치를 기반으로 노드의 출력을 계산하는 함수입니다. 활성화 함수가 *비선형* 인 경우 몇 개의 노드만 사용하여 사소한 문제를 해결할 수 있습니다 .
        
        시그모이드, 하이퍼볼릭 탄젠트→ vanishing gradient 문제
        
        vanishing gradient 문제 : 역전파 과정에서 입력층으로 갈수록 기울기가 작아지는 현상. 입력층에 인접한 층에서의 가중치가 업데이트되지 않음.
        
        →RELU함수는 기존 다른 함수들의 문제점을 해결함. 그래서 주로 RELU함수를 이용.
        
        RELU→입력값이 음수인 뉴런들을
        

최종 결과물과 실제 결과를 비교해 오차를 구하고 역전파를 통해 가중치를 업데이트

stochastic gradient descent : 확률적 경사 하강법

참고 자료

[[머신러닝 - 이론] 딥러닝 - 다층 퍼셉트론 구조, 다층 퍼셉트론의 학습 방법(Deep Learning - Multi Layer Perceptron structrue, MLP Learning method)](https://hi-guten-tag.tistory.com/53)

[03. 기본 신경망 - 다층 퍼셉트론(MLP : Multi Layer Perceptron)](https://wikidocs.net/227541)

[07-06 과적합(Overfitting)을 막는 방법들](https://wikidocs.net/61374)

[loss function (손실 함수), cost function (비용 함수)](https://wikidocs.net/120077)

# 합성곱 연산(Convolution)

- 합성곱 수학적 정의

합성곱(合成-), 또는 콘벌루션(convolution)은 **하나의 함수와 또 다른 함수를 반전 이동한 값을 곱한 다음, 구간에 대해 적분하여 새로운 함수를 구하는 수학 연산자**이다. 

![img](https://github.com/IMS-STUDY/AI-Study/assets/127017020/dbd88406-24e0-4f26-84e4-7be7a0a68fdd)

시간의 흐름에 따라 g(x)가 이동하면서 f(x)를 평균적으로 얼마나 변화시키는지 나타내는 것으로 볼 수 있음.

- 핵심 원리

입력 데이터, 필터가 필요.

입력 데이터는 주로 정방행렬이 이용됨

필터의 값은 가중치라고 볼 수 있음. 주로 3x3, 5x5를 사용함(홀수)

입력데이터와 필터의 원소 값의 곱을 더한 결과는 피쳐맵의 원소가 됨

피쳐맵은 합성곱 연산의 결과임

합성곱 연산은 입력된 이미지의 특징을 찾기 위해 필터를 씌우는 과정임. 이미지는 너비, 높이, 채널로 구성되는데 흑백 이미지의 경우 채널은 1이고 컬러이미지의 경우 채널이 3임.

합성곱 연산을 진행하면 데이터 손실이 발생해 점점 크기가 줄어들어 결국 합성곱 연산을 적용하지 못하게 됨.

이를 해결하기 위해 패딩을 이용함.

패딩은 입력 데이터 주변에 0을 채워넣어 손실을 없애는 방법임

- 밸리드 패딩 : 패딩을 추가하지 않음
- 풀패딩 : 입력데이터가 mxm, 필터가 nxn일 때 n-1픽셀의 패딩을 추가
- 세임패딩 : 위의 내용에서 (n-1)/2픽셀 만큼의 패딩을 추가

스트라이드는 필터의 이동 크기임. 스트라이드 만큼 필터를 이동시켜 연산을 수행함.

합성곱 연산에는 가중치와 편향을 추가할 수 있음

채널의 수와 커널의 수는 일치해야함.

- 풀링 : 피처맵의 크기를 줄이거나 특정 데이터를 강조하는 연산.
    
    -최대풀링
    
    -평균풀링
    

합성곱 연산은 이미지 처리에서 필터 연산에 해당한다.

참고 자료

[합성곱 신경망 — 응용수학  documentation](https://compmath.korea.ac.kr/appmath2022/ConvolutionNN.html)

[11-01 합성곱 신경망(Convolution Neural Network)](https://wikidocs.net/64066)

[But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)

[[TensorFlow 2.x 강의 12] 컨볼루션 개념](https://www.youtube.com/watch?v=63Y4tP_soXc)

[컨볼루션 (Convolution) 개념](https://blog.naver.com/PostView.naver?blogId=beyondlegend&logNo=222256886960)

[Convolution 연산 이해하기 for CNN](https://velog.io/@minchoul2/Convolution-연산-이해하기-for-CNN)

[컨볼루션 연산에 대해](https://kionkim.github.io/2018/06/08/Convolution_arithmetic/)

[컨볼루션(합성 곱) 에 대한 직관적 이해](https://people-analysis.tistory.com/264)

데이터 전처리 : 레코드를 기반으로 필드를 조작하는 것

1. 데이터 발생
2. 수집
3. DBMS
4. 읽기
5. 데이터프레임
6. 데이터 분석
- 코랩
    
    구글코랩 접속→노트북 생성 나머지는 파이썬과 같다
    
- 파이토치
    
    

[인공지능 7/4](https://www.notion.so/7-4-377925fcbc2246ccb25e480e5a26aaaa?pvs=21)