# 2주차 과제

## 활성화 함수란

신경망에서 입력 신호의 가중치 합을 출력 신호로 변환하는 함수다.

비선형성을 가지고 있으며, 입력에 대한 비선형 변환을 통해 신경망이 다양한 종류의 복잡한 함수를 학습할 수 있게 한다.

활성화 함수는 주로 각 뉴런의 출력 값을 결정하고 신경망의 비선형성을 제공하는 역할을 수행한다.

선형 계층은 아무리 layer가 쌓여도 결국 하나의 선형 구조로 표현이 가능하다. 하지만 비선형 함수를 사용하면 깊은 layer를 쌓아 특성을 추출할 수 있다.

- 계단함수
    - 초기에 사용하던 함수로, 연속성을 표현할 수 없다.
    
    ![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/548f9207-e551-4f44-b6e4-159c04daf1b0)
    
- 시그모이드(sigmoid)
    - 가중치가 바뀌는 과정에 연속성을 부여한다.
    - 무한대의 실수 값을 0과 1사이의 값으로 변환하는 역할을 한다.
    - 입력 값의 절댓값이 커질수록 기울기가 0에 수렴한다.
    
    ![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/4f21dfc6-1c5a-4707-8e66-e06cea59ba7c)
    
- Tahn
    - 쌍곡선 함수라고도 한다.
    - 시그모이드 함수에 비해 기울기가 작아지지 않는 구간이 커서 양수와 음수 모두에서 학습 효율이 뛰어나다.
    - 심층 신경망에서 sigmoid함수의 역할이 필요하다면 대안으로 사용되기도 한다.
    - 기울기가 0으로 수렴하는 구간이 존재하여 기울기 문제가 존재한다.
    - 범위는 -1과 1 사이다.
    
    ![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/3ea25191-1cba-4353-bca8-27d4b65e239f)
    
- ReLu
    - 입력값이 음수면 0을 출력하고 양수면 그대로 출력한다.
    - sigmoid의 기울기 소실 문제를 해결한 함수다.
    - 연산이 쉽고 Layer를 깊게 쌓을 수 있다는 장점이 있다.
    - 범위는 0부터 무한대이다.

![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/cffdbd4f-3c42-429f-868e-18df24ca3eed)

- softmax
    - 클래스에 속할 확률을 계산하는 함수이므로 은닉층에서 사용하지 않고 출력층에서 사용된다.
    - 다중 분류 모델에서 주로 사용된다.
    - 입력값을 지수함수로 취하고, 정규화한다.(출력의 합이 1)
    

활성화 함수가 선형 구조라면, 미분 과정에서 상수가 나오게 되므로 학습이 진행되지 않는다.

신경망은 기울기를 이용하여 최적의 값을 찾는 것인데 기울기가 0에 수렴하게 되면 성능이 떨어지게 되므로 기울기가 0으로 수렴하는 함수는 잘 사용하지 않는다.

# K-Fold Cross Validation

- 데이터를 Train set과 Test set으로 나누고, Train set내부에서  K개의 Fold로 나눈다. Valid set으로 선택된 Fold를 제외한 K-1개의 Fold를 Train set으로 사용하고, Valid set을 통해 평가한다. 이를 Train set 내에서 Valid set을 변경해가며 반복한다. 이후 최적의 모델을 선정하여 Test set을 통해 평가한다.

![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/e15612bf-4f49-4325-922b-53a4e773d124)