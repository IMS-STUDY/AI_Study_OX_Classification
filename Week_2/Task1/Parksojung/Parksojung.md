# 7/8

*2주-1*

# [이론]

## *Acitvation Function*

1. ***acitvation function란? - 활성화 함수***

![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/5707d60e-0ebb-44f4-b066-962f41b1f71a)

- 입력 신호의 가중치 합을 출력 신호로 변환하는 함수
- 입력 신호의 가중치 합이 활성화를 일으키는지 정의하는 역할
- 주로 각 뉴런의 출력 값을 결정하고 신경망의 비선형성을 제공하는 역할을 함
- 비선형을 가지고 있음 → 입력에 대한 비선형 변환을 통해 신경망이 다양한 종류의 복잡한 함수를 학습할 수 있게 함

*[목적]*

비선형 관계, 비선형 특성을 고려하기 위해 비선형 활성화 함수를 활용함

[비선형 함수를 사용해야 하는 이유]

선형 함수란, 출력이 입력의 상수 배만큼 변하는 함수. 1개의 곧은 직선

비선형 함수란, 선형이 아닌 함수. 직선 하나로는 그릴 수 없는 함수.

신경망에서 선형 함수를 이요하면 신경망의 층을 깊게하는 의미가 없어짐.

층을 아무리 깊게 해도 은닉층이 없는 네트워크로도 똑같은 기능을 할 수 있기 때문.
(y(x)= h(h(h(x)))=ax)

따라서 층을 쌓기 위해서는 활성화 함수 중에서도 비선형 함수를 사용해야 함.

1. ***종류, 특징***
    - ***sigmoid***
        
        ![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/e6ddacce-07e6-4113-bffe-898fb5bc2c12)
        
        - 실수 값을 입력받아 0~1 사이의 값으로 압축함
        - 큰 음수 값일수록 0에 가까워지고 큰 양수 값일수록 1이 됨
        
        (단점)
        
        - 기울기 소멸 문제가 발생함 → 기울기가 입력이 0일때 가장 크고, |x|가 클수록 기울기는 0에 수렴함. 이는 역전파 중에 이전의 기울기와 현재 기울기를 곱하면서 점점 기울기가 사라지게 됨. 그러면 신경망의 습 능역이 제한되는 포화(Saturation)가 발생.
        - 시그모이드 함수값은 0이 중심(zero-centered)이 아님 → 학습 속도가 느려짐
    
    - ***tanh***
        
        ![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/ade4d550-e687-4f5e-b4ac-ffc2616a4ccb)
        
        - 실수 값을 입력 받아 -1 ~ 1 사이의 값으로 압축함
        - 시그모이드를 두배해주고 -1한 값과 비슷함
        
        ![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/52483a91-cd3e-47a5-acc9-68f3a3d87ba1)
        
        - 결과값이 -1 ~ 1 사이 → 중심(zero-centered)에 있어서 지그재그가 덜하여 시그모이드에 비해 최적화를 잘함
        - 하지만 기울기 소멸 문제가 있음
    - ***relu***
        
        ![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/d4272762-3a10-49be-aa81-d864c9458ab0)
        
        - 가장 많이 사용하는 활성화 함수
        - 입력이 0이 넘으면 → 입력을 그대로 출력
        입력이 0 이하이면 → 0 출력
        - 양수 부분에서는 포화(saturate)가 발생하지 않음
        - exp 연산이 없어서 빠름
        - 하지만 0 중심 (zero-centered)가 아니라 지그재그 문제가 있음(최적화가 오래걸림)
    - ***softmax***
        
        ![Untitled 5](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/73d793e1-e32c-453b-a203-e06f70086638)
        
        - 시그모이드와 비슷한 모양
        - 실수 값을 입력 받아 0~1 사이로 변환하여 출력하지만 출력 값들의 합이 1이 되도록 하는 함수
        - 다중 분류의 최종 활성화 함수로 사용(소프트맥스 함수를 통해 얻은 확률 값들은 서로 종속)
        - 분류될 클래스가 n개라고 하면, n차원의 벡터를 입력 받아 각 클래스에 속할 확률을 추정함 → 값들이 확률 분포를 이룸
        

## *k-fold*

- *k-fold란?*

k 개의 fold를 만들어서 진행하는 교차검증

***[과정]***

![Untitled 6](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/4fa417d0-a771-43bb-a1be-06ea1bcd3704)

1. 전체 데이터를 train/test 데이터로 나눔
2. train 데이터를 다시 train/valid 데이터로 K개의 fold로 나눔
3. Split1에서 Fold 1을 valid, Fold 2 ~ Fold 5를 train으로 사용하여 train으로 모델을 훈련한 후, valid로 평가함
4. 마찬가지로, Split k에서 Fold k를 valid, Fold k를 제외한 나머지 fold를 train으로 사용하여 train으로 훈련 후 valid로 평가함
5. Split k개에서 각각 성능을 평가하므로 총 k개의 성능 결과가 나오는데, 이 k개의 평균을 낸 것이 해당 학습 모델의 성능임

![Untitled 7](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/876a45a3-4c89-4c5d-ad17-1981c07fdc07)

[사용이유]

- 총 데이터 갯수가 적은 데이터 셋에 대해 정확도 향상 가능
- 이는 기존에 Training / Validation / Test 3개의 집단으로 분류하는 것보다, Training과 Test로만 분류할 때 학습 데이터 셋이 더 많기 때문
- 과적합 방지 가능

# [실습]

## *OX_Classification*

## *MLP로 분류 진행하기*

- *OX_Dataset 불러오기*
- *OX_image 전처리*
- *MLP로 진행시 k-fold적용 , 미적용 각각 정확도 산출*