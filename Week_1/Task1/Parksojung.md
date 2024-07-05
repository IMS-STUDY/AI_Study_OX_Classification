# 7/1

1. **딥러닝 기초**

![Untitled](https://github.com/IMS-STUDY/AI-Study/assets/127017020/9c583df9-1647-42a7-a8e0-155d83e84f7b)

- 딥러닝 개요
    - **Deep Learning 란?**
    
    : Deep learning is a subset of [machine learning](https://www.ibm.com/topics/machine-learning) (데이터와 알고리즘을 사용하여 AI가 인간이 학습하는 방식을 모방하고 점차 정확도를 향상 시킬 수 있도록 )that uses multilayered [neural networks](https://www.ibm.com/topics/neural-networks)(인간의 뇌와 유사한 방식으로 결정을 내리는 기계 학습 프로그램 / 모델), called deep neural networks, to simulate the complex decision-making power of the human brain. 
    
    : 인간의 두뇌에서 영감을 얻은 방식으로 데이터를 처리 할 수 있도록 컴퓨터를 가르치는 인공지능 방식.
    :머신 러닝의 한 종류로, 다중 신경망을 이용함(deep neural network 심층 신경망)
    
    - 머신러닝 vs 딥러닝
    
    ![Untitled 1](https://github.com/IMS-STUDY/AI-Study/assets/127017020/d468c29b-fb47-402e-b285-8f2f05fb3fea)
    
    **머신러닝**은 주어진 데이터를 인간이 먼저 처리함. 사람이 먼저 컴퓨터에 특정 패턴을 추출하는 방법을 지시하고, 그 후 컴퓨터가 스스로 데이터의 특징을 분석하고 축적함. 그리고 축적된 데이터를 바탕으로 문제를 해결. 
    
    **딥러닝**은 머신러닝에서 사람이 하던 패턴 추출 작업이 생략되고, 컴퓨터가 스스로 데이터 기반으로 학습할 수 있도록 정해진 신경망을 컴퓨터에게 주고 학습을 수행함.
    
    ---
    
    [참고 사이트]
    
    [https://www.databricks.com/kr/glossary/artificial-neural-network](https://www.databricks.com/kr/glossary/artificial-neural-network)
    
    - ANN ?
    
    : 인공신경망, Artificial Neural Network
    
    : 뉴런과 시냅스로 구성된 인간의 뇌 신경망을 노드와 링크로 모형화한 네트워크 형태의 지도학습 모형.
    
    → “가중치를 적용한 방향성 그래프”
    
    다층 ANN은 복잡한 분류나 회귀 작업을 해결하는 데 쓰임.
    
    ---
    
    - **MLP란?**
    
    : 다층 퍼셉트론, multi-layer perceptron
    
    :  perceptron은 가장 단순한 유형의 인공 신경망. 데이터를 선형적으로 분리할 수 있는 경우에만 효과가 있고, 대개 이진법 예측을 하는데 쓰임
    
    MLP는 딥러닝 방법 중 하나로 완전히 연결된 다층 신경망임. 하나 이상의 인공 뉴런이나 노드 계층으로 이루어져 있음.
    
    [형태]
    
    퍼셉트론으로 이루어진 층(layer) 여러 개를 순차적으로 붙여 놓은 형태
    
    ![Untitled 2](https://github.com/IMS-STUDY/AI-Study/assets/127017020/2060db06-1f5f-4449-8309-119e861a3300)
    

입력에 가까운 층을 아래에 있다 하고, 출력에 가까운 층을 위에 있다 하면 신호는 *아래→위* 방향으로 계속 움직임(단방향, 네트워크를 학습하기 위해 역전파를 사용). 입력 노드를 제외한 각 노드에는 비선형 활성화 함수가 있음. 

 MLP에서는 인접한 층의 퍼셉트론간의 연결은 있어도 같은 층의 퍼셉트론끼리의 연결을 없음. 또, 한번 지나간 층으로 다시 연결되는 피드백도 없음. 

[사용되는 부분]

MLP는 지도 학습이 필요한 문제를 해결하는 것 뿐만 아니라 전산 신경 과학 및 병렬 분산 처리에 대한 연구에도 널리 사용됨. 응용 프로그램에는 음성 인식, 이미지 인식, 기계 번역이 포함됨.

---

- 합성곱(convolution) 연산

의미적 → 두 함수를 서로 곱해서 합함

![Untitled 3](https://github.com/IMS-STUDY/AI-Study/assets/127017020/dbcdc2f3-c8cc-46eb-a17e-2cca2799a4eb)

위 식은 두 연속함수 f, g를 convolution 하는 식임.

1) 합성곱을 위해서는 두 함수 중 하나를 reverse(반전)시켜야 함.
위 식에서는 g의 변수 타우(𝜏) 앞쪽에 마이너스가 붙어 있음 → g를 reverse 시켰다는걸 알 수 있음

1) 반전시킨 함수를 shift(전이) 시킴.
위 식에서는 g를 t만큼 이동 시켰음을 알 수 있음.

1) 이동 시킨 함수 g를 함수 f 와 곱한 결과를 하나씩 기록함.
이때 변수 타우(𝜏)를 변화 시키며 결과를 쭉 기록하는 것을 convolution 이라함.
이동 시키면서 겹치는 각 지점의 함수 값을 곱하고 그 곱을 더해 새로운 함수를 만들어냄. 이를 통해 두 함수의 상호작용을 나타내는 (새로운)함수가 생성됨.

[응용 분야]

Convolution은 신호처리 분야에서 가장 많이 사용되는 연산 중 하나. 

기본적으로 입력 신호에 특정 형태의 필터(커널이라고도 불림)를 씌워 원하는 결과를 얻어내는 방식으로 사용이 됨.
그 필터는 다른 신호가 될 수도 있고 고정된 값이 될 수 도 있음. 딥러닝에서는 보통 고정된 값의 필터를 사용함. 그리고 그 필터의 값들이 가중치가 됨.
함수를 shift할때 →여기선 stride라고도 함. 한 필터에서 다음 필터로 갈 때 몇 칸을 띄어서 가는지를 의미.

이때 Neural Networks 에서 사용되는 Convolution 이 CNN임(Convolutional Neural Networks)

CNN: 데이터로부터 직접 학습하는 딥러닝의 신경망 아키텍처.

[작동 방식]

수십, 수백개의 계층을 가질 수 있으며, 각 계층(필터)은 영상의 서로 다른 특징을 검출함. 각 훈련 영상에 서로 다른 해상도의 필터가 적용되고, convolution된 각 영상은 다음 계층의 입력으로 사용됨. 

[어디에 사용?]

영상에서 객체, 클래스, 범주 인식을 위한 패턴을 찾을 때 유용함. 오디오, 시계열 및 신호 데이터를 분류하는데도 매우 효과적.

![Untitled 4](https://github.com/IMS-STUDY/AI-Study/assets/127017020/687ee8b3-bdab-4048-a201-7c64d3e6982c)

Layers in convolutional neural network (CNN)

- 비용함수란? 손실함수란?

일단 기계 학습 모델이 실제로 작동하려면 높은 수준의 정확성이 필요함. 하지만 모델이 얼마나 옳고 그른지 어떻게 계산하는가? 이것 때문에 비용함수 & 손실함수가 등장함. 
즉, 모델을 올바르게 판단하는 데 사용되는 기계 학습 매개변수인 비용 함수는 모델이 입력 매개 변수와 출력 매개 변수 간의 관계를 얼마나 잘 추정했는지 이해하는데 중요함.

**비용함수 cost function : 손실함수를 제곱함 or 평균 등의 형식으로 정의,  training set에 있는 모든 샘플에 대한 손실함의 합**

: 입력과 출력 간의 관계를 찾는 데 모델이 얼마나 잘못 되었는지 측정하는데 사용됨. 모델이 얼마나 나쁘게 동작하고 예측하는지 알려줌.

→ 내가 만든 모델이 실제 정답과 얼마나 다른지를 측정하는 수단

실제 값 - 모델 예측 값 = 에러

비용함수가 크다 ⇒ 모델의 정확도가 낮다

[예시]

![Untitled 5](https://github.com/IMS-STUDY/AI-Study/assets/127017020/98a663a6-a519-4991-9bfc-cd5c4130dc85)

회색 원이 실제 값, 빨간 선이 예측 값(모델)이라 했을 때, 파란선이 에러.
파란선의 총합이 클수록 에러가 큼.

이 오류를 줄이기 위해 사용되는 방법이 2가지 있음.

[1. 각 error에 절댓값을 씌운 뒤 그것을 합산하는 방법]

[2. 각 error를 제곱한 뒤 그것을 합산하는 방법] → 이 방법을 많이 사용함

![Untitled 6](https://github.com/IMS-STUDY/AI-Study/assets/127017020/8a58aa8f-557b-46c2-a75f-ba0d433e0fab)

각 error를 제곱한 뒤 더하는 방식으로 모델의 전체적인 오류값을 하나의 수식으로 표현 가능 → 이차식의 형태가 됨(제곱했기 때문).
이 이차식을 평면에 그리면 위와 같은 포물선이 만들어지고, 이런 그래프를 비용함수(cost function)이라 함. 이때 error 가 최소인 지점이 우리가 가고자하는 최적점이 됨.

**손실함수 loss function : 개별적인 차이를 정의**

: 머신러닝이나 딥러닝 모델이 예측한 값과 실제 값 사이의 error를 측정하는 함수.

[비용함수 vs 손실함수]

| 비용함수 | 손실함수 |
| --- | --- |
| more general.
sum of loss functions over your training set plus model complexity penalty (training set에 대한 손실 함수의 합과 모델 복잡성 패널티) | a function defined on a data point.
predict, label, and measures the penalty |
|  |  |
|  |  |

- 과적합(overfitting)이란?

데이터에 맞춰서 모델을 학습 시켰는데, 이 모델이 현재 가지고 있는 데이터에서만 잘 작동하는 것. 그러니까 모델을 학습시킬 때 현재 데이터에 포함 되어있는 noise 까지 학습해서 정작 예측 성능은 저하가 된 것. 그래서 현재 데이터가 아닌 새로운 데이터를 넣어 예측을 하려고 하면 잘 작동하지 않음. 대부분 응용에서 발생하며 학습 데이터가 적거나 문제가 어려울수록 과적합의 정도가 심해짐.

![Untitled 7](https://github.com/IMS-STUDY/AI-Study/assets/127017020/15fb6735-5abe-44d2-bc61-6a426634318e)

![Untitled 8](https://github.com/IMS-STUDY/AI-Study/assets/127017020/d82195f4-2f64-4961-8223-e34f21f9dc83)

 [그래프에 대한 설명]

underfitting → 학습 오차와 테스트 오차가 같이 감소함

overfitting → 학습 오차는 감소하지만, 테스트 오차는 증가함

즉, 우리의 목적은 학습을 통해 예측 모델의 underfitting된 부분을 제거하면서 overfitting이 발생하기 직전에 학습을 멈춤. 머신 러닝에서는 overfitting을 방지하기 위해 여러 방법이 연구 되었으며 일반적으로 validation set을 이용하여 overfitting이 일어났는지 판별함.

- 방지 방법

딥러닝 모델에서 과적합이 발생되었을 때 해결할 수 있는 방법은 다음과 같음

1) 데이터 양 늘리기
데이터 양을 늘려서 데이터의 일반적인 패턴을 학습하여 과적합을 방지함.
하지만 데이터의 양이 적을 경우, 의도적으로 기존의 데이터를 조금씩 변형하고 추가해서 데이터의 양을 늘리기도 함(데이터 증식 또는 증강, data augmentation). 

2) 모델의 복잡도 줄이기
 인공 신경망 모델의 복잡도를 줄이는 것

3) 드롭아웃(drop out) 적용하기
학습 과정에서 신경망의 일부를 사용하지 않는 방법.

![Untitled 9](https://github.com/IMS-STUDY/AI-Study/assets/127017020/8dbee19f-3eb8-4390-a11b-66cd24d41e16)

4) 가중치 규제(Regularization) 적용하기
간단한 모델은 적은 수의 매개변수를 만듦. 복잡한 모델을 좀 더 간단하게 만들기 위해 가중치 규제를 적용함. 즉 학습 데이터를 낮추고 평가 데이터 정확도를 높이는 것. 모델이 학습 데이터에 지나치게 의존하지 않도록 패널티를 부과하는 것. 
- L1 규제: 가중치 w들의 절대값 합계를 비용 함수에 추가함.
- L2 규제: 모든 가중치 w들의 제곱합을 비용 함수에 추가함.

---

[참고] [https://yeouido-developer.tistory.com/entry/데이터-전처리의-종류](https://yeouido-developer.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC%EC%9D%98-%EC%A2%85%EB%A5%98)

[https://yssa.tistory.com/entry/Big-Data-데이터-전처리](https://yssa.tistory.com/entry/Big-Data-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC) 

1. **데이터 전처리**
- 데이터 전처리란?

데이터 분석을 하기 위해 수집한 데이터를 분석에 적합한 형태로 가공하는 과정

데이터 전처리 과정을 통해 필요없는 데이터를 제거하고, 결측치나 이상치를 처리하여 데이터의 질을 향상시킬 수 있음.(raw data → clean data)

- 데이터 전처리 목적성

데이터에 잡음, 이상치, 부적합, 결측치, 중복 등이 많으면 데이터를 적절하게 분석하는데 어려움이 있어 필요없는 값들은 삭제하고 부족한 값들은 적절한 값으로 채워 넣을 필요가 있음. 데이터 전처리를 어떻게 하냐에 따라 결과가 달라지기 때문에 중요한 과정임.

- 잡음(noise): 측정 과정에서의 오류 값
- 이상치(outlier): 일반적인 데이터에 비해 다른 특성을 가지는 튀는 값
- 부적합(inconsistent): 모순된 잘못된 데이터 값
- 결측치(missing value): 누락된 데이터 값
- 중복(duplicated): 중복되는 데이터
- 데이터 전처리 절차 및 방법

![Untitled 10](https://github.com/IMS-STUDY/AI-Study/assets/127017020/68d53ba1-2325-494f-be33-cc07dfb3ec14)

1) 데이터 수집: 데이터를 수집함

2) 데이터 정제 (data cleaning): 형식의 일관성을 유지하고 적합한 포맷으로 변환함(잡음, 부적합, 결측치 등의 읽을 수 없는 요소 제거함)

[데이터 정제 절차]

- 데이터 오류 원인 분석

| 원인 | 오류 처리 방법 |
| --- | --- |
| 결측값(Missing value) | -평균값, 중앙값, 최빈값 같은 중심 경향값 넣기
-랜덤에 의해 자주 나타나는 값을 넣는 분포기반 처리 |
| 노이즈(Noise) | - 일정 간격으로 이동하면서 주변보다 높거나 낮으면 평균값으로 대체
- 일정 범위를 중간값으로 대체 |
| 이상값(Outlier) | - 하한보다 낮으면 하한값으로 대체
- 상한보다 높으면 상한값으로 대체 |
- 데이터 정제 대상 선정
    - 모든 데이터를 대상으로 정제 활동을 하는 것이 기본
    - 특히 데이터 품질 저하의 위험(적합성, 신뢰성 )이 있는 데이터에 대해서는 더 많은 정제 활동을 해야 함
    - 원천 데이터의 위치를 기준으로 분류한다면 내부 데이터보다 외부 데이터가 품질 저하 위협에 많이 노출되어있고,
    정형 데이터보다는 비정형과 반정형 데이터가 품질 저하 위협에 많이 노출되어 있음.
        
        
        | 종류 | 설명 |
        | --- | --- |
        | 정형 데이터(Structured Data) |  미리 정해 놓은 형식과 구조에 따라 저장되도록 구성하여 고정된 필드에 저장된 데이터 |
        | 비정형 데이터(Semi-Structured Data) | 정의된 구조가 없는 동영상 파일, 오디오 파일, 사진, 보고서 등 정형화 되지 않은 데이터 |
        | 반정형 데이터(Unstructured Data) | 데이터의 구조 정보를 데이터와 함께 제공하는 파일 형식의 데이터로, 데이터의 형식과 구조가 변경될 수 있는 데이터 |
- 데이터 정제 방법 결정
    - 데이터 정제는 오류 데이터값을 정확한 데이터로 수저하거나 삭제하는 과정
    - 정제 여부의 점검은 정제 규칙을 이용하여 위반되는 데이터를 검색하는 방법
    - 노이즈나 이상값은 비정형 데이터에서 자주 발생→ 데이터 특성에 맞는 정제 규칙을 수립하여 점
    
    [정제 방법]
    
    | 방법 | 설명 |
    | --- | --- |
    | 삭제 | + 오류 데이터에 대해 전체 또는 부분 삭제
    - 무작위적인 삭제는 데이터 활용에 문제를 일으킬 수 있음 |
    | 대체 | + 오류 데이터를 평균값, 최빈값, 중앙값으로 대체
    - 오류 데이터가 수집된 다른 데이터와 관계가 있는 경우 유용할 수 있지만 그렇지 않은 경우 왜곡이 발생함 |
    | 예측값 | + 회귀식 등을 이용한 예측값을 생성하여 삽입
    - 예측값을 적용하기 위해서는 정상 데이터 구간에 대해서도 회귀식이 잘 성립되어야 함 |

[데이터 정제 기술]

- 데이터 일관성 유지를 위한 정제 기법
    
    
    | 기법 | 설명 |
    | --- | --- |
    | 변환(Transform) | 다양한 형태로 표현된 값을 일관된 형태로 변환하는 작업 |
    | 파싱(Parsing) | 데이터를 정제 규칙을 적용하기 위한 유의미한 최소 단위로 분할하는 과정 |
    | 보강(Enhancement) | 변환, 파싱, 수정, 표준화 등을 통한 추가 정보를 반영하는 작업 |
- 데이터 정제 기술
    
    
    | 기술 | 설명 |
    | --- | --- |
    | ETL (Extract, Transform, Load) | 수집 대상 데이터를 추출, 가공하여 데이터 웨어하우스 및 데이터 마트에 저장하는 기술 |
    |  Map Reduce | - 모든 데이터를 키-값 쌍으로 구성하여 데이터를 분류
    - 데이터를 추출하는 Map 기술, 추출한 데이터를 중복 없게 처리하는 Reduce 기술로 구성 |
    | Spark/Storm | 맵리듀스를 기반으로 성능을 개선한 것. 배치 처리 모두 가능, 기계 학습,  라이브러리 지원 가능 |
    | CEP (Complex Event Processing) | 실시가능로 발생하는 이벤트 처리에 대한 결과값을 수집하고 처리하는 기술 |
    | Pig | 대용량 데이터 집합을 분석하기 위한 플랫폼 |
    | Flume | 로그 데이터를 수집하고 처리하는 기법 |

3) 데이터 통합 (data integration): 다양한 소스에서 얻은 데이터를 단일 형식으로 변경한 후 동일한 단위나 좌표로 변환함

4) 데이터 축소 (data reduction): 일반적인 데이터는 크기가 너무 커서 한번에 분석이 불가능 → 축소 시킨 데이터를 사용하는 것이 효과적

5) 데이터 변환 (data transformation): 정규화, 집계 등을 통해 데이터 형식이나 구조를 변환

- 데이터 전처리 기초 문법

```python
# 데이터에 결측치가 있는지 확인
#(컬럼별 결측치 개수)=(전체 데이이터 건수) - (각 컬럼 별 값이 있는 데이터 수)
data = pd.read_csv(csv_file_path)
data.head()
len(data) #전체 데이터 건수
len(data) - data.count() # 컬럼별 결측치 개수

# 결측치 여부를 True, False로 반환
DataFrame.isnull() 
# 결측치가 있는 데이터 제거
df.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
=>
labels : 삭제할 레이블명
axis : {0 : index / 1 : columns} labels인수를 사용할경우 지정할 축
index : 인덱스명을 입력해서 바로 삭제
columns : 컬럼명을 입력해서 바로 삭제
level : 멀티인덱스의 경우 레벨을 지정 가능
inplace : 원본을 변경할 여부. True일경우 원본이 변경.
errors : 삭제할 레이블을 찾지 못할경우 오류. ignore할 경우 존재하는 레이블만 삭제됨.

# 결측치 다른 값으로 대체
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
=>
axis : {0: index / 1: columns} 결측치 제거를 진행 할 레이블
how : {'any' : 존재하면 제거 / 'all' : 모두 결측치면 제거} 제거할 유형
			포함만 시켜도 제거할지, 전무 NA여야 제거할지 정함
tresh : 결측값이 아닌 값이 몇 개 미만일 경우에만 적용시키는 인수
		예를들어, tresh값이 3이라면 결측값이 아닌 값이 3개 미만일 경우에만 dropna메서드를 수행
subset : dropna메서드를 수행할 레이블을 지정
inplace : 원본을 변경할지의 여부

# 중복된 데이터 확인
df.duplicated(subset=None, keep='first')
subset : 특정 열만 가능. list 등등..
keep : {first : 위부터 검사 / last : 아래부터 검사} 검사 순서
		first일 경우 위부터 확인해서 중복행이 나오면 True를 반환하, last일 경우 아래부터 확인

# 이상치 제거, 분석 ->IQR(Interquartile range): 사분위범위수를 사용해서 이상치를 찾음
		IQR=Q3-Q1 (Q1−1.5∗IQR보다 왼쪽에 있거나
		Q_3 + 1.5*IQRQ3+1.5∗IQR 보다 오른쪽에 있는 경우 이상치라고 판단
# 데이터 프레임 생
# IQR 계산
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q

# 이상치 제거
data_cleaned = 
	data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

```

---

1. pytorch 다루기

[https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-01-introduction/](https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-01-introduction/)

- 환경 설정
- pytorch 구성요소, 문법, 특징

[pytorch란?]

[https://truman.tistory.com/213](https://truman.tistory.com/213)

- 파이썬 기반의 과학 연산 패키지
- 넘파이를 대체하면서 GPU를 이용한 연산이 필요한 경우 사용함
- 최대한의 유연성과 속도를 제공하는 딥러닝 연구 플랫폼이 필요한 경우 사용

→“강력한 GPU 가속이 적용되는 파이썬으로 된 텐서와 동적 신경망”

[특징]

- 텐서플로보다 간결해서 쉽게 사용 가능
- 학습 및 추론 속도가 빠르고 다루기 쉬움
- Define-by-Run(동적 계산 그래프를 생성하는 방법) 프레임워크

[Tensor 텐서]

- 파이토치의 기본 단위
- 다차원 배열을 처리하기 위한 데이터 구조
- numpy와 ndarray와 거의 같은 API를 지니고 있음
- GPU를 사용한 계산도 지원
- 어떤 데이터 형의 텐서이건 torch.tensor라는 함수로 작성 가능

[패키지 구성요소]

- torch
    - main namespace임. tensor 등의 다양한 수학 함수가 패키지에 포함되어 있음
    - NumPy 와 같은 구조를 가지고 있어 넘파이와 비슷한 문법 구조를 가짐
- torch.autograd
    - 자동 미분을 위한 함수가 포함되어 있음
    - 자동 미분의 on, off를 제어하는 enable_grad 또는 no_grad나 자체 미분 가능 함수를 정의 할 때 사용하는 기반 클래스인 Function 등이 포함됨
- torch.nn
    - 신경망을 구축하기 위한 다양한 데이터 구조나 레이어가 정의되어 있음
    - CNN, LSTM, 활성화 함수(ReLu), loss 등이 정의되어 있음
- torch.optim
    - SGD(Stochastic Gradient Descent) 등의 파라미터 최적화 알고리즘 등이 구현되어 있음
- torch.utils.data
    - Gradient Descent 계열의 반복 연산을 할 때, 사용하는 미니 배치용 유틸리티 함수가 포함되어 있음(데이터 조작 등 유틸리티 기능 제)
- torch.onnx
    - ONNX(Open Neural Network eXchange) 포맷으로 모델을 export 할 때 사용함
    - ONNX는 서로 다른 딥러닝 프레임워크 간에 모델을 공유할 때 사용하는 새로운 포맷

[문법]

[https://truman.tistory.com/213](https://truman.tistory.com/213)

| 문법 | 사용 방법 |
| --- | --- |
| import torch | 라이브러리 불러오기 |
| x = torch.empty(5,4)
print(x) | 빈 텐서 생성(행:4, 열:5) |
| torch.rand(5,6)
torch.one(3,3) | → 랜덤한 값으로 구성한 텐서 생성
→ 1로 구성한 텐서 생성 |
| l = [13,4]
torch.tensor(l)
r = np.array([4,56,7])
torch.tensor(r) | → 리스트를 텐서로 변환

→ nparray를 텐서로 변환 |
| x.size()
type(x) | → 텐서 사이즈 확인
→ 텐서 타입 확인 |
| x+y
torch.add(x,y)
y.add(x) | x,y인 텐서가 있다고 하면,
텐서의 연산() |