# 2 - CNN

# CNN

## Convolution Neural Network

[https://www.researchgate.net/publication/336805909/figure/fig1/AS:817888827023360@1572011300751/Schematic-diagram-of-a-basic-convolutional-neural-network-CNN-architecture-26.ppm](https://www.researchgate.net/publication/336805909/figure/fig1/AS:817888827023360@1572011300751/Schematic-diagram-of-a-basic-convolutional-neural-network-CNN-architecture-26.ppm)

- 개요
    - 합성곱 연산을 이용한 신경망 구조 알고리즘
    - 인간의 시신경 구조를 모방
    - 이미지, 비디오를 처리하는 비전 분야에서 유용하게 사용됨
- 구조
    - 이미지 정보(3차원 텐서 등)를 입력으로 받아서
    - 합성곱(Convolution) 레이어를 통과
        - 이때 합성곱 연산을 통하여 이미지의 특징을 추출
    - 이후 합성곱 결과에 ReLU(Rectified Linear Unit)를 적용
        
        [https://t1.daumcdn.net/cfile/tistory/246B094F57F226C036](https://t1.daumcdn.net/cfile/tistory/246B094F57F226C036)
        
        - 활성함수 종류 중 하나, Sigmoid 함수를 대체(Gradient Vanishing 해결)
        
        // Gradient Vanishing: 기울기 소실, 레이어가 많아질수록 역전파 과정에서 기울기가 매우 작아지는 현상
        
        - 입력값이 0보다 작으면 0을, 0보다 크면 입력값을 그대로 리턴하는 함수
    - 이후 풀링 레이어를 통과
        - 데이터의 차원을 감소시켜 입력의 매개변수를 줄임
        - 크게 두가지로 분류, 최대 풀링과 평균 풀링
        - 복잡도가 낮아져 과적합을 방지
    - 이후 **완전 연결 계층**(Fully Connected Layer) 을 통과
        - 한 층의 모든 뉴런이 다음 층의 모든 뉴런과 연결된 상태
        - 1차원 배열의 형태를 입력값으로 받아서 이미지를 분류
        - 다차원 배열이 1차원으로 평탄화되므로 일부 정보가 소실되는 단점
            
            → 공간적 구조, 픽셀 간 거리 등과 같은 정보 소실
            
            → 그래서 이 계층을 통과하기 전 합성곱, 풀링 과정을 거치는 것
            
            ![Untitled](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/34130ab7-3a69-4321-810d-b7acef789351)

            

## 합성곱

![Untitled 1](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/2e61a539-6c04-497a-9423-b362ea514bb2)

![Untitled 2](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/5510ee8c-6c4e-4a25-8adc-4be8982c15c2)

출처:  https://www.youtube.com/watch?v=KuXjwB4LzSA

- 입력 데이터에 필터(커널)을 적용하여 결과(특성 맵, Feature Map)를 추출하는 연산 작업
- 필터는 가중치 역할을 함
- 입력 데이터에 필터를 일정 간격(스트라이드, Stride)만큼 이동시키면서 계산
- 대응하는 원소끼리 곱한 후 총합을 구함

![Untitled 3](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/718c7756-72fe-4aa4-8d73-287ebb7bcf41)

## Padding

https://ardino.tistory.com/40

![Untitled 4](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/85eceb7f-8b17-498a-8520-132445cfe334)

- 합성곱 연산을 수행하기 전 입력 데이터의 주변을 특정 값으로 채워 확장하는 것
- 주로 0으로 채워넣는 Zero Padding을 사용
- 합성곱 연산을 할 때마다 연산 결과의 사이즈가 작아지는 것을 방지
- Valid Padding: 패딩을 추가하지 않은 경우
- Full Padding: 입력 데이터의 모든 원소가 동일한 합성곱 연산 참여 횟수를 가지도록 패딩을 적용
    - 예시
        
        ![Untitled 5](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/98af6e0a-4747-4aa2-88d2-ac3d2b6467c5)
        
        위 (4x4) 데이터에 (3x3) 필터를 적용하여 합성곱 연산을 하려는 경우
        
        입력 데이터의 각 원소의 합성곱 연산 참여 횟수가 동일
        
- Same Padding: 합성곱 연산 결과의 크기가 입력 데이터의 크기와 동일하도록 패딩을 적용
    - 예시
        
        ![Untitled 6](https://github.com/IMS-STUDY/2024_Summer_AI_Study/assets/127017020/c6211d4b-6609-4a7c-a993-04c5c198a72a)
        위 (4x4) 데이터에 (3x3) 필터를 적용하여 합성곱 연산을 하려는 경우
        
        합성곱 연산을 수행한 결과도 (4x4) 크기를 가짐
        

## Pooling

![Untitled](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Sh9e6Hzx8ZcOinuLvy8Fmw.png)

- 데이터의 사이즈(공간적 크기)를 축소시키는 연산과정
- 복잡도가 감소하여 과적합을 방지하는 효과가 있음
- Max Pooling: 특정 영역에서 최대값만을 찾아 나타냄
- Average Pooling: 특정 영역에서 평균을 구하여 나타냄

## conv(n)d [n = 1, 2]

https://leeejihyun.tistory.com/37

- Pytorch 나 Keras 등과 같은 딥러닝 라이브러리에서 합성곱 연산을 할 때 사용하는 메소드
- n은 입력데이터의 차원을 결정
    
    → 정확하게는 합성곱 연산의 진행 방향(1d의 경우 한 방향, 2d의 경우 가로, 세로 두 방향)의 차이
    

# MLP 구현(이론)

Pytorch 사용

1) 필요한 패키지 import

→ numpy, matplotlib, torch, torch.nn, torch.optim 등등

2) 데이터셋 로드

→ torchvision의 built-in dataset 등 다른 모듈에서 제공해주는 데이터셋을 사용하거나

→ 데이터셋을 직접 불러와도 사용 가능

예시) O/X 분류 모델에서 직접 그린 그림을 사용하고자 하는 경우:

→ PyTorch의 Dataset 클래스를 상속받는 새로운 클래스를 생성하여 사용자 정의 데이터셋을 구현할 수 있음

- 예시 코드(https://wikidocs.net/194918)
    
    ```python
    **import** torch
    **from** torch.utils.data **import** Dataset
    **class CustomImageDataset**(**Dataset**):
    		**def __init__**(self, file_paths, labels, transform=None):
            """
            커스텀 이미지 데이터셋 클래스의 생성자입니다.
            Args:
                file_paths (list): 이미지 파일 경로의 리스트
                labels (list): 이미지 레이블의 리스트
                transform (callable, optional): 이미지에 적용할 전처리 함수
            """
            self.file_paths = file_paths
            self.labels = labels
            self.transform = transform
            
    		**def __len__**(self):
            #데이터셋의 전체 샘플 개수를 반환합니다.
            **return** len(self.file_paths)
            
        **def __getitem__**(self, idx):
    			   """
    	        인덱스에 해당하는 샘플을 가져옵니다.
    
    	        Args:
    	            idx (int): 샘플의 인덱스
    	
    	        Returns:
    	            image (torch.Tensor): 이미지 데이터의 텐서
    	            label (torch.Tensor): 이미지 레이블의 텐서
    	        """
    	        # 이미지 파일을 불러옴
    	        image = Image.open(self.file_paths[idx])
    	
    	        # 이미지에 전처리 함수를 적용 (예: Resize, RandomCrop, ToTensor 등)
    	        **if** self.transform **is not** None:
    			      image = self.transform(image)        
    	        # 이미지 레이블을 텐서로 변환        
    	        label = torch.tensor(self.labels[idx])
    	        **return** image, label
    ```
    

3) 모델 만들기

→ torch.nn.Module 또는 torch.nn.Sequential 클래스를 상속한 클래스를 정의

Module 클래스는 forward() 메소드 정의 필요, Sequential 클래스는 자동으로 레이어들을 순차적으로 연결

→ 모델의 순전파 동작을 정의(nn.Conv2d, nn.Flatten, nn.ReLU 등 메서드 활용)

→ 다양한 레이어들을 추가하고 조합하여 모델을 구축

4) 모델 학습시키기

→ 모델에 데이터를 집어넣어 첫 학습 진행

→ 이후 해당 결과를 이용하여 역전파 과정을 거쳐 가중치 업데이트

→ 업데이트된 가중치를 이용하여 반복 학습 진행

5) 모델 평가하기

→ accuracy, f1_score 등 다양한 지표로 모델의 성능을 평가

→ PyTorch에서 제공하는 model.dval() 메소드도 있음(평가 모드, 모델을 고정시킴)