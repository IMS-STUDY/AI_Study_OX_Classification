# 인공지능 1주차 - 1

추가 일시: 2024년 7월 1일 오후 3:22
강의: 인공지능

-정보를 찾을때 영어로 검색하기

-링크를 남기면서 필기하기

# 딥러닝 기초

비선형 변환기법의 조합을 통해 높은 수준의 추상화를 시도하는 기계학습 알고리즘의 집합

## ANN

사람의 뇌에서 뉴련들이 자극들을 받고 자극이 임계값을 넘어 결과 신호를 전달하는 신경망 원리와 구조를 모방하여 만든 기계학습 알고리즘

들어오는 자극은 input data, 
임계값은 가중치인 weight,
자극에 의해 행동하는 것은 output

신경망은 데이터를 받는 입력층, 출력을 담당하는 출력층, 입력층과 출력층 사이의 레이어들은 은닉층이 존재한다.

![Untitled](https://github.com/IMS-STUDY/AI-Study/assets/127017020/4d212122-3dfd-4b56-92eb-5c21ab52e580)

은닉층에서 활성화함수를 사용하여 최적의 가중치(w)와 편향(bias)를 찾아야한다.

### 문제점→기기의 성능발전

1. 파라미터의 최적값을 찾기 어렵다(gradient가 뒤로갈수록 점점작아진다)
2. 학습시간이 너무 느림(은닉층이 많아서)

## DNN

ANN모델에서 은닉층을 많이 늘려 학습의 결과를 향상시키는 방법

## CNN

기존방식은 데이터에서 지식을 추출해 학습이 이루어졌지만 CNN은 데이터의 특징을 추출하여 특징들의 패턴을 파악하는 구조이다. 데이터를 추출하는Convolution과 레이어 사이즈를 줄이는Pooling과정을 통해 알고리즘을 구성한다.

![Untitled 1](https://github.com/IMS-STUDY/AI-Study/assets/127017020/24ec0301-2960-4057-8726-61d733d912f8)

![Untitled 2](https://github.com/IMS-STUDY/AI-Study/assets/127017020/577c0883-0d56-4820-bb93-8463bcf51a18)

![Untitled 3](https://github.com/IMS-STUDY/AI-Study/assets/127017020/a974c432-29b9-41c1-a616-15a08f18014e)
## RNN

반복적이고 순차적인 데이터 학습에 특화된 인공신경망의 한 종류로써 내부의 순환구조가 들어가있다. 순환구조를 이용하여 과거의 학습을 가중치를 통해 현재학습에 반영한다. 현재의 학습과 과거의 학습을 연결가능하게 하고 시간에 종속된다는 특징이 있다.이를 통해서 음성관련된 부분에 사용된다.
이전시점에서의 은닉층의 메모리 셀에서 나온값을 자신의 이후 자신의 입력으로 사용하는 재귀적 활동

![Untitled 4](https://github.com/IMS-STUDY/AI-Study/assets/127017020/31734c3a-3857-412e-ac27-612d30ea9b3b)

## MLP란?

은닉층 개수를 늘린 깊은 신경망
인접한 층의 퍼셉트로간의 연결은 있어도 퍼셉트론끼리의 연결은 없다. 또한 한번 지나간 층으로 다시 연결되는 피드백도 없다.

![Untitled 5](https://github.com/IMS-STUDY/AI-Study/assets/127017020/12b3d307-d9b9-4322-b414-e96b0cb85a04)

### 다른모델과의 차이점

**CNN vs MLP: 데이터를 어떻게 추출하느냐에 정해져있다. CNN은 Convolution과 Pooling을 이용해 추출하고 MLP는 데이터값을 바로 추출한다**

![Untitled 6](https://github.com/IMS-STUDY/AI-Study/assets/127017020/5fe4ecab-bc82-4ebd-8dd9-0bff17455723)

MLP와 DNN의 차이는 MLP가 DNN의 하위개념으로 봐서 별 차이가 없다
하지만 이에 반대되는 의견도 있다.

## 합성곱연산

데이터가 필터를 통과하고 피처맵이 되는 형태이다
원본 데이터의 특징을 찾기위해 필터를 씌우는 과정이다. 여기서 필터와 유사한 이미지의 영역을 강조하는 특성 맵을 출력하여 다음층으로 전달한다(여기서 필터가 이동하는 간격은 스트라이라고 한다, 출력데이터가 줄어드는걸 방지하기 위해 패딩을 이용한다)

![Untitled 7](https://github.com/IMS-STUDY/AI-Study/assets/127017020/ba0b2121-6b79-45f9-b642-2f2b7886cd48)

풀링레이어는 피처맵의 크기를 줄이거나 특정데이터를 강조하는 용도로 사용한다.
최대, 최소, 평균을 이용하여 구한다

![Untitled 8](https://github.com/IMS-STUDY/AI-Study/assets/127017020/f323fd46-b355-4df9-9a2a-dba6cdb43fe9)

## 손실함수란?

실제값에 비해 만든 모델이 얼마나 잘 예측했는지 판단하는 함수
제곱오차, 절대오차, logloss가 있다

## 비용함수란?

원래의 값과 가장 오차가 작은 가설함수를 도출하기 위해 사용되는 함수
모델 학습시 사용하는 데이터는 여러값으로 이루어져있다. 따라서 손실함수의 평균으로 비용함수를 구한다
제곱오차의 평균인 MSE 절대오차의 평균인 MAE logloss의 평균인 Binary Cross-entropy가 있다

![Untitled 9](https://github.com/IMS-STUDY/AI-Study/assets/127017020/a537f76b-76b2-4bef-9d27-3f00a4079586)

## 과접합이란?

알고리즘이 학습데이터에 과하게 적합하거나 정확하게 일치하여 결과모델이 학습데이터가 아닌 다른 데이터에서 정확하 예측을 생성하거나 결론을 도출하지 못하는 현상이다.
과하게 적합한것은 모델이 데이터의 노이즈를 기억하는 경우를 말한다

![Untitled 10](https://github.com/IMS-STUDY/AI-Study/assets/127017020/27be17e9-a3bb-44f0-b7c3-baf07bd54efd)
### 과적합 방지 방법

1. 더많은 훈련자료를 이용한다
2. 관련성이 있는 데이터만 사용한다
3. 정규화를 해준다(특징의 영향을 줄인다, 아래w를 0으로 변경하는 경우)

![Untitled 11](https://github.com/IMS-STUDY/AI-Study/assets/127017020/11aacea2-3c37-4118-b3e5-dca716e7d1af)

# pytorch다루기

## 환경설정

GPU가 없는 환경에서 파이토치를 설치하는건 쉽지않다. 따라서 colab으로 파이토치를 이용한다

코랩에서 파이토치 설치 
[런타임] - [런타임 유형 변경 ] - [하드웨어가속기를 GPU로 바꿔주기]통해서 GPU환경으로 변경해야한다. 이후 코드작성하여 설치

!pip3 install torch
!pip3 install torchvision

![Untitled 12](https://github.com/IMS-STUDY/AI-Study/assets/127017020/e205dd16-0a6b-4d3b-bdcb-6f9d4aa1b97e)

## pytorch구성요소, 문법, 특징

### 특징

GPU에서 텐서 조작 및 동적 신경망 구축이 가능한 프레임워크
GPU: 연산속도를 빠르게 하는 역할, 병렬 연산에서는 CPU의 속도보다 훨씬 빠름
텐서: 파이토치의 데이터형태이다. 다차원 테이블을 담은 수학 객체라고 할 수 있다.
         배열, 행렬과 유사한 특수한 자료구조, 연산가속을 위한 H/W에서 사용가능한 장점
동적신경망: 훈련을 반복할 때마다 네트워크 변경이 가능한 신경망→학습중 은닉층 추가, 제거등 모델의 네트워크 조작이 가능하다.

### 구성요소

torch: 메인 네임스페이스, 텐서 등의 다양한 수학 함수가 포함

torch.autograd: 자동 미분 기능을 제공하는 라이브러리

torch.nn: 신경망 구축을 위한 데이터 구조나 레이어 등의 라이브러리

torch.multiprocessing: 병럴처리 기능을 제공하는 라이브러리

torch.optim: SGD(Stochastic Gradient Descent)를 중심으로 한 파라미터 최적화 알고리즘 제공

torch.utils: 데이터 조작 등 유틸리티 기능 제공

torch.onnx: ONNX(Open Neural Network Exchange), 서로 다른 프레임워크 간의 모델을 공유할 때 사용

### 문법

텐서생성

이상한 값들이 들어간 텐서

```python
x = torch.empty(4,2) # (4,2) 크기의 텐서생성

> tensor([[7.1685e-35, 0.0000e+00],  
          [5.0447e-44, 0.0000e+00],
          [       nan, 0.0000e+00],
          [1.3788e-14, 3.6423e-06]])
```

무작위로 초기화한 텐서

```python
x = torch.rand(4,2) # (4,2) 크기의 0~1 사이 랜덤값 텐서생성

> tensor([[0.9879, 0.7454],
          [0.0494, 0.3015],
          [0.6462, 0.3983],
          [0.8883, 0.9433]])
```

사용자가 입력한 텐서

```python
x = torch.tensor([1.6,2,4,5])

> tensor([1.6000, 2.0000, 4.0000, 5.0000])
```

0 / 1으로 채워진 텐서

```python
x = torch.zeros(4,2,dtype=torch.long)
y = torch.ones(3,2,dtype=torch.float)

> tensor([[0, 0],
         [0, 0],
         [0, 0],
         [0, 0]])
    
> tensor([[1., 1.],
         [1., 1.],
         [1., 1.]])
```

'같은크기'를 가지는 무작위 텐서 생성

```python
x = torch.randn_like(x,dtype=torch.float)

> tensor([[-0.6504,  2.9375],
          [-0.1826, -0.0519],
          [ 0.3183,  1.0060],
          [ 0.6136,  0.7064]])
```

텐서 이용하는 함수

- x.type() : 텐서의 타입 알려줌 (큰 종류만, torch.FloatTensor)
- x.dtype : 텐서의 타입 알려줌 (자세히, torch.float32)
- x.shape : 텐서의 shape 알려줌
- x.size() : 텐서의 shape 알려줌

텐서연산

```python
torch.abs(a)
torch.min(a)
torch.max(a)
torch.mean(a)
torch.std(a) # 표준편차
torch.prod(a) # 곱
torch.ceil(a) # 올림
torch.floor(a) # 내림
```

텐서행렬연산(기본연산은 스칼라연산이다)

```python
x = torch.tensor([[1,2,3],[4,5,6]],dtype=float)
y = torch.randn_like(x,dtype=float)

# 스칼라연산
print(x*y)
> tensor([[  1.4303,  -0.8309,  -0.1673],
          [ -3.0072, -10.9429,   1.8235]], dtype=torch.float64)
        
# 행렬연산
print(torch.matmul(x,y_t))
> tensor([[  0.4321,  -4.2172],
          [  3.3092, -12.1266]], dtype=torch.float64)
```

기타연산

1. x.type(torch.float) #타입변환

```python
x = torch.tensor(1)
x = x.type(torch.float)
```

1. torch.unique(텐서) #집합형태로 만듬(중복 삭제)

```python
x = torch.tensor([1,2,3,4,1,2,3,1])
x_set = torch.unique(x)

> tensor([1, 2, 3, 4, 1, 2, 3, 1])
> tensor([1, 2, 3, 4])
```

1. torch.clamp(텐서,범위1,범위2) 
값들이 범위안에있고 벗어나면 지정값으로로 지정

```python
x = torch.tensor([1,2,3,4,1,2,3,1])
x_clamp = torch.clamp(x,1.5,2.5)

> tensor([1, 2, 3, 4, 1, 2, 3, 1])
> tensor([1.5000, 2.0000, 2.5000, 2.5000, 1.5000, 2.0000, 2.5000, 1.5000])
```

1. 데이터.max(dim=0) 행열에 따른 최대/최소
- 데이터.max(dim=0) : 열기준 argmax
- 데이터.max(dim=1) : 행기준 argmax
- 데이터.min(dim=0) : 열기준 argmin
- 데이터.min(dim=1) : 행기준 argmin

```python
x = torch.rand(3,4)
arg = x.max(dim=0)

print(x)
print(arg)

> tensor([[0.7002, 0.5021, 0.8545, 0.2372],
         [0.0192, 0.5245, 0.7978, 0.5102],
         [0.1585, 0.4971, 0.7144, 0.5225]])
> torch.return_types.max(
> values=tensor([0.7002, 0.5245, 0.8545, 0.5225]),
> indices=tensor([0, 1, 0, 2]))
```

1. 데이터.argmax(축 번호) 가장큰값을 가진 인덱스번호를 보여줌

```python
x = torch.rand(4,2)
print(x)
print(x.argmax(0))
print(x.argmax(1))

> tensor([[0.2154, 0.5701],
         [0.2725, 0.8342],
         [0.3470, 0.2540],
         [0.2194, 0.9083]])
> tensor([2, 3])
> tensor([1, 1, 0, 1])
```

# 데이터 전처리

## 데이터 전처리란?

데이터 분석을 위해 수집한 데이터를 분석에 적합한 형태로 가공하는 과정
불필요한 데이터를 제거하고 결측치나 이상치를 처리하여 데이터의 질을 향상시킨다.
데이터 발생->수집->DBMS->읽기->데이터프레임->데이터분석

## 데이터 전처리 목적성

원시 데이터로부터 유용한 정보를 추출, 머신러닝 모델이 이를 효과적으로 학습하고 일반화 가능하도록 데이터 품질을 향상시킴

## 데이터 전처리 절차 및 방법

1. 데이터셋 확인: 변수 유형, 변수 간의 관계 및 분포
2. 결측값과 이상값 처리

결측유형
완전 무작위 결측: 응답을 빠뜨린 경우 랜덤하게 발생하는 결측
무작위 결측: 결측이 다른 변수와 연관있지만 결과분포에는 영향 안주는 경우
비무작위 결측: 결측값이 결과에 영향을 미치는 경우

결측처리방법

삭제: 완적 무작위 결측, 데이터수가 충분히 많은경우 효율적

대체: 완전무작위 결측이 아닌 경우 사용

이상값 검출방법

분산: 일반적으로 정규분포 97.5% 이상 또는 2.5% 이하의 값을 이상값이라고 한다
우도: 우도함수의 우도 값 범위 밖의 가능성이 낮은 범위에서 발견되는 값을 이상값이라고 한다
근접 이웃 기반 이상치 탐지: 정사값들과 떨어진 위치에 있는 이상값을 탐지하는 거리를 이용한 방법이다
밀도를 기반으로 한 탐지: 상대적 밀도를 고려해 이상치를 탐지하는 방법으로 밀도 있는 데이터에서 떨어져 위치한 데이터를 이상값으로 간주
군집: 비슷한 개체를 묶어 정상값이 포함된 특정 군집(묶음)에 속하지 않는 경우 이상치로 판별하는 거리를 이용한 방법이다
사분위수: 25%(Q1)과 75%(Q3)을 활용하여 이상치를 판단. IQR(= Q3-Q1, 사분범위) * 1.5를 벗어나는 경우 이상치로 판단한다

이상값 처리 방법: 삭제, 대체, 스케일링, 정규화하처 이상값을 처리한다

1. feature  engineering: 기존 변수사용, 정보 추가, 기존 데이터 보완

산입: 누락데이터가 있으면 해당 열을 날림

예외값 처리하기: 기준에 안맞는 예외값들을 제거한다

Bin으로 묶기: 근처의 값들을 하나의 범주로 묶는다

로그 변형: 왜도와 첨도가 높은경우 상관성을 파악하기 어렵워 이 값들을 정규분포에 맞게 변형시켜야 한다.

One Hot Encoding: 텍스트로 된 데이터가 있고 없고를 0, 1로 표현

## 데이터 전처리 기초 문법

파이썬의 pandas를 이용하여 데이터를 전처리한다

파일 쓰기

![Untitled 13](https://github.com/IMS-STUDY/AI-Study/assets/127017020/f292d6b4-9a70-4bfc-867f-a0ce32928588)

파일 읽기

![Untitled 14](https://github.com/IMS-STUDY/AI-Study/assets/127017020/16ef8ede-163c-45f6-aa8b-5af1a4990f41)

reindex() 메소드: 인덱스 재배열할 때 사용(아래 예시는 재배열로 발생한 NaN값을 0)

![Untitled 15](https://github.com/IMS-STUDY/AI-Study/assets/127017020/1cbdcd95-7cba-4f9f-ad79-3ae624e4b0e6)

sort_index() 메소드 : 행 인덱스를 기준으로 정렬(오름차순이 기본)

![Untitled 16](https://github.com/IMS-STUDY/AI-Study/assets/127017020/49b25d79-e654-45b6-9a52-59c1d30674e3)

fillna()메소드를 통해 값을 치환

![Untitled 17](https://github.com/IMS-STUDY/AI-Study/assets/127017020/1d9c6160-a2e7-488d-8f1f-e03386f14f83)

dropna 메서드로 누락데이터 삭제

![Untitled 18](https://github.com/IMS-STUDY/AI-Study/assets/127017020/271b4b04-4726-4111-80f4-fda11dca7f73)

# 코랩사용법

## 기본 단축키

실행(Enter)

Ctrl: 해당셀을 실행
Shift: 해당셀 실행후 다음셀 대기

Esc: 눌러 값이 입력이 안되는 상태를 만든다

이후 a,b로 새로운 셀 만들기 z로 삭제, 방향키로 오가기

Ctrl + MM 마크다운 형태로 변경
Ctrl + MY 코드셀 형태로 변경

# 용어

활성화함수: 입력신호를 출력신호로 변환하는 함수
→ 입력신호를 가중치와 편향을 이용해 계산하여 출력값을 보여준다(시그모이드, 렐루등)
비선형으로 이뤄지는 이유는 신경망 층을 깊게하는 의미가 없어진다

# 출처

https://ebbnflow.tistory.com/119

https://happy-obok.tistory.com/55

https://wikidocs.net/22886

https://deepestdocs.readthedocs.io/en/latest/004_deep_learning_part_2/0040/

https://stats.stackexchange.com/questions/315402/multi-layer-perceptron-vs-deep-neural-network

https://ardino.tistory.com/39

https://datawith.tistory.com/94

https://roytravel.tistory.com/103

https://velog.io/@regista/비용함수Cost-Function-손실함수Loss-function-목적함수Objective-Function-Ai-tech

https://box-world.tistory.com/6

https://www.ibm.com/kr-ko/topics/overfitting

https://m.blog.naver.com/snova84/223294035159?referrerCode=1

https://m.blog.naver.com/snova84/223294035159?referrerCode=1

https://didu-story.tistory.com/67

https://resultofeffort.tistory.com/77

https://velog.io/@changwoo7463/PyTorch1

https://chaheekwon.tistory.com/entry/데이터-전처리의-개념과-중요성

https://velog.io/@kphantom/2.-데이터전처리란

https://velog.io/@ttogle918/데이터-전처리-이론

https://magoker.tistory.com/118