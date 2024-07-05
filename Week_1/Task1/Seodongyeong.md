# 1 - 딥러닝

# 딥러닝 개요

[https://blog.bismart.com/en/difference-between-machine-learning-deep-learning](https://blog.bismart.com/en/difference-between-machine-learning-deep-learning)

![https://i0.wp.com/semiengineering.com/wp-content/uploads/2018/01/MLvsDL.png?resize=733,405&ssl=1](https://i0.wp.com/semiengineering.com/wp-content/uploads/2018/01/MLvsDL.png?resize=733,405&ssl=1)

- **딥러닝 개요**
    - **정의**
        - 머신러닝의 한 종류로, 상대적으로 다른 학습 방법들보다 인간이 사고하는 방식에 더 가깝게 학습하는 방법
        - 인공신경망(Neural Network) 중에서도 심층 신경망(Deep Neural Network)을 사용하여 학습하는 방법이 딥러닝
    - **역사**
        - 초기에는 인간의 뉴런을 모방한 Perceptron 등장(Single - Layer)
            
            → 인공신경망(Artificial Neural Network) 의 최초 구현
            
            → 입력층 - 출력층 구성
            
            ![Untitled](https://github.com/IMS-STUDY/AI-Study/assets/127017020/9c713a49-5adb-4d52-98bd-37cea7c973cb)
            
            → xor게이트 연산 불가능 - 대부분의 현실 문제 해결 불가능
            
        - 위 문제를 해결한 MLP(Multi - Layer Perceptron) 등장
            
            ![Untitled 1](https://github.com/IMS-STUDY/AI-Study/assets/127017020/8798cd0e-193a-4e20-9ace-57638411be1d)
            
            → 중간에 *은닉층을 추가하여 다중 계층 구조 형성
            
            *은닉층 == 입력 데이터로부터 복잡한 특징이나 패턴을 추출하는 역할을 수행
            
        - 은닉층이 2개 이상인 신경망  - DNN(Deep Neural Network)
            
            ![Untitled 2](https://github.com/IMS-STUDY/AI-Study/assets/127017020/531f4132-c7ce-44be-9200-f9e5c6a96ad8)
            
            → 이 신경망을 사용한 머신러닝을 딥 러닝이라고 칭함
            
            → 이 DNN을 기반으로 한 다른 알고리즘들이 개발됨( CNN, RNN 등)
            
    
- **합성곱 연산**
    - 개요
        
        ![Untitled 3](https://github.com/IMS-STUDY/AI-Study/assets/127017020/2e9950b8-4d53-4ce5-8ef3-2ed97357fec5)
        
        ![https://blog.kakaocdn.net/dn/c7wid5/btrbgnXVx1j/3c44KYGuqHYJ6X2d2QE6pk/img.gif](https://blog.kakaocdn.net/dn/c7wid5/btrbgnXVx1j/3c44KYGuqHYJ6X2d2QE6pk/img.gif)
        
        
        - 두 연속함수를 합성하는 연산
        - 하나의 함수를 반전, 전이한 후 두 함수를 곱한 값을 적분하여 새로운 함수를 구하는 연산
    - 과정
        1. 두 함수 중 하나에 -1을 곱하여 반전시킨다.
        2. 반전시킨 함수를 임의의 변수 t만큼 전이시킨다.
        3. 두 함수를 서로 곱한 결과를 적분한다. 이때 임의의 변수 𝜏를 변화시키면서 적분한다.
        
        [https://wikimedia.org/api/rest_v1/media/math/render/svg/38a7dcde9730ef0853809fefc18d88771f95206c](https://wikimedia.org/api/rest_v1/media/math/render/svg/38a7dcde9730ef0853809fefc18d88771f95206c)
        
    - 특징
        - 이 합성곱 연산은 교환법칙, 결합법칙, 분배법칙이 성립한다.
        - 주로 f함수를 원래 가지고 있던 정보(신호, 행렬 등), g함수를 필터, 가중치 등으로 표현하여 f함수를 목적에 맞게 변환, 필터링 할 때 사용
- **비용함수 / 손실함수**
    - 비용함수(Cost Function), 손실함수(Loss Function)는 서로 같은 의미로, 모델이 예측한 결과와 실제 결과가 얼마나 일치하는지를 나타내는 기준함수를 말한다.
    - 차이점:
        - Loss Function은 좁은 범위, 단일 데이터셋에 대한 결과
            
            ex) Squared Error, Absolute Error
            
        - Cost Function은 전체 데이터셋에 대한 결과, 즉 개별 Loss Function의 평균
            
            ex) Mean Squared Error, Mean Absolute Error
            
            +) 평균 이외에도 Loss Function의 합을 사용하기도 함
            
    
- **과적합, 방지 방법**
    - 과적합(Overfitting) 개요
        
        ![Untitled 4](https://github.com/IMS-STUDY/AI-Study/assets/127017020/2d37e405-94f9-45f4-aae4-a53612c1841c)
        
        - 모델이 Training Data에만 높은 정확도를 보이는 현상
        - Training Data의 노이즈까지 학습하게 되어 처음 보는 데이터(Validation Data)에 대해서는 제대로 예측을 하지 못함
        - 주로 모델이 지나치게 복잡한 경우 발생
    - 방지 대책
        - 하이퍼파라미터를 조정하여 모델의 복잡도를 적정 수준으로 조절
        - 불균형 데이터 이슈인지 확인하고 부족한 데이터 보강(Data Augmentation)하기
- 참조한 문서
    
    https://wikidocs.net/24958
    
    https://ko.wikipedia.org/wiki/합성곱
    
    https://velog.io/@hyesoup/머신러닝-overfitting-개념과-해결-방법-feat.-기울어진-운동장
    

# PyTorch

- 환경설정
    
    https://pytorch.org/get-started/locally/ → 공식 다운로드 페이지 접속
    
    ![Untitled 5](https://github.com/IMS-STUDY/AI-Study/assets/127017020/8dd24dec-69f4-4106-b97c-497c0d64f294)

    
    
    - 기본으로 현재 하드웨어에 맞는 설정 선택, 필요에 따라서 옵션을 변경
    - 이후 맨 아래쪽의 명령어를 환경에 맞게 실행
        - Windows, pip 설치의 경우 cmd창에 입력 후 실행
    - 기타 자주 쓰이는 패키지들(numpy, pandas, matplotlib 등) 사전 설치 권장
    - Google Colab 사용 시 따로 설치 불필요, 바로 `import` 해서 사용
- 특징
    - 딥러닝에 특화되어 있는 오픈소스 프레임워크로, Python 기반 라이브러리이나 자바, C++을 제한적으로 지원하기도 함
    - PyTorch에서는 주로 텐서(Tensor)라는 단위를 사용
        - 텐서란 다차원 배열로, 1차원 텐서 == 벡터, 2차원 텐서 == 행렬에 대응되며 3차원 이상부터는 n차원 텐서로 부름
        - NumPy의 ndarray와 유사:    NumPy식 인덱싱, 슬라이싱이 모두 가능
            
            → 단, ndarray와는 다르게 GPU로도 연산이 가능하여 딥러닝 및 병렬연산에 강함
            
        - 대표적으로 이미지 처리의 경우 3차원 텐서(width, height, batch size)를 사용
    - 역전파(Back Propagation, 학습 결과를 바탕으로 가중치를 재조정) 과정에서 유용하게 사용되는 자동 미분 엔진인 autograd 모듈 제공
    - 이외에도 최적화 알고리즘을 구현할 때 도움을 주는 torch.optim 모듈 등 다양한 모듈들을 제공
    
    → 더 자세한 내용은 공식 문서로: https://pytorch.org/docs/stable/torch.html  
    
- 기본 메소드
    
     →`import torch` 필요
    
    - 텐서 초기화(생성) 관련
        - torch.tensor(data)  # data로부터 직접 텐서를 생성, 자료형은 자동 감지
        - torch.from_numpy(np_array)  # ndarray로부터 텐서를 생성
        - # shape == 텐서의 사이즈를 나타내는 튜플
        - torch.rand(shape)  # 무작위 값을 넣은 텐서를 생성
        - torch.ones(shape)  # 모든 원소가 1인 텐서를 생성
        - torch.zeros(shape) # 모든 원소가 0인 텐서를 생성
    - 텐서 속성 관련
        - torch.shape  # 텐서의 사이즈를 리턴
        - torch.dtype  # 텐서의 자료형을 리턴
        - torch.device  # 텐서가 저장된 장치를 리턴(cpu, cuda(gpu) 등)
    
- 참조 문서
    
    https://wikidocs.net/52460
    
    https://tutorials.pytorch.kr/beginner/basics/intro.html
    
    https://www.ibm.com/kr-ko/topics/pytorch
    

# 데이터 전처리

- 정의
    
    → 전처리(Preprocessing), 즉 데이터를 사용하기 전에 미리 입맛에 맞게 가공하는 과정을 뜻함
    
- 목적
    
    → 한정된 데이터로 최대의 학습 효율을 끌어내기 위함
    
    → 머신러닝의 경우 대부분 데이터의 수는 다다익선, 그러나 현실은 항상 이상적이지 않음
    
- 방법
    
    → 데이터의 결측치, 이상치를 제거(pandas의 .dropna() 등)
    
    → 연속형 데이터의 경우 정규화, 표준화 과정 적용
    
    (정규화의 경우 각 feature의 범위를 특정 값 사이로 조정, 표준화의 경우 데이터가 표준정규분포를 따르게끔 조정)
    
    → 사람이 직접 연관이 없는 데이터를 drop
    
- 예시
    - NumPy:
        
        → unique() 메소드로 중복된 값 제거
        
    - Pandas:
        
        → drop() 메소드로 dataframe에서 행 또는 열을 삭제
        
        → dropna() 메소드로 결측치(NaN) 제거
        
    - Scikit-Learn:
        
        → sklearn.preprocessing.MinMaxScaler() 메소드로 데이터 정규화
        

# Google Colab

- 구글 코랩이란?

![Untitled 6](https://github.com/IMS-STUDY/AI-Study/assets/127017020/4036ca63-23c4-4c3b-9536-0b16350e986c)

→ 구글에서 호스팅하는 Jupyter Notebook 서비스로, 웹에서 파이썬 코드를 작성 및 실행 가능

→ 사용자의 리소스를 사용하지 않으며, 무료로(제한적) 구글에서 제공해주는 CPU, GPU를 사용할 수 있음

- 사용법
    1. https://colab.research.google.com/ 에 접속, 구글 계정만 있다면 바로 사용 가능
    2. 사전에 본인 목적에 맞는 런타임 연결 설정(CPU, GPU, TPU 등)
        
        → 단, CPU의 경우 싱글코어만 할당해주기 때문에 개인 pc보다 느릴 수 있음
        
    3. Jupyter Notebook 스타일을 사용하기 때문에 해당 플랫폼의 사용법과 대부분 동일(단축키 등 일부 차이점 존재)
        
        → 다른 IDE와 크게 다른점은 소스파일 전체를 컴파일하지 않고 셀 단위로 컴파일이 가능하다는 점, 따라서 중간에 실행 결과를 확인하기 쉬움
        
        → 셀 앞에 [ ], 대괄호 부분이 현재 실행된 순서를 나타냄
        
        → 현재 변수 상태 등이 모두 이 순서를 따르므로 자주 확인 필요
        
        → 코드 셀에서 마크다운 셀로 전환이 가능하여 약간의 문서화가 가능함
        
    
    유용한 단축키)
    
    | 단축키 | 설명 |
    | --- | --- |
    | A | 선택한 셀 위에 코드 셀 추가 |
    | B | 아래에 코드 셀 추가 |
    | Ctrl + M + D | 셀 삭제 |
    | Ctrl + Enter | 현재 셀 실행 |
    | Shift + Enter | 선택한 셀 실행 후 다음 셀 선택 |
    | Ctrl + F9 | 모든 셀들을 차례대로 실행 |
    | Ctrl + M + M | 현재 셀을 마크다운 셀로 전환 |
    | Ctrl + M + Y | 현재 셀을 코드 셀로 전환 |