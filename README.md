## 파일 구조
```bash
mini_DriveGAN/
├─ data/
│  ├─ frames/          # 학습에 사용하는 주행 이미지 프레임
│  └─ labels.csv       # 이미지 파일명과 steering 값 정보
├─ dataset.py          # 데이터셋 로딩 및 (현재 프레임, 조향값, 다음 프레임) 구성
├─ model.py            # Encoder, Transition Model, Decoder로 이루어진 1단계 모델
├─ train.py            # 학습 실행 코드
├─ utils.py            # 예측 결과 이미지 저장 함수
└─ prepare_from_zip.py # 원본 zip 파일에서 필요한 이미지와 labels.csv 생성
```

[Used Kaggle Dataset](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning?resource=download&select=self_driving_car_dataset_jungle)

# Mini DriveGAN Stage 1

이 프로젝트는 DriveGAN의 아이디어를 이해하기 위해 만든 간단한 1단계 실험입니다.  
전체 논문을 그대로 재현하려 한 것이 아니라, **현재 주행 이미지와 조향값(steering)을 이용해 다음 이미지를 예측하는 구조**를 직접 구현해보는 것이 목표였습니다.

이 실험은 **DriveGAN: Towards a Controllable High-Quality Neural Simulation** 논문을 읽은 뒤 시작했습니다.  
논문을 보면서 단순히 이미지를 생성하는 것이 아니라, **현재 장면과 차량의 행동을 바탕으로 다음 장면이 어떻게 변하는지 예측하는 구조**가 인상적이었습니다.  
그래서 논문 전체를 바로 따라 하기보다는, 관련 개념을 조금이라도 더 이해하고자 그 핵심 개념에 가까워지기 위해 작은 형태의 모델을 직접 구현해보았습니다.

## 프로젝트 설명

이 모델은 현재 프레임을 입력으로 받아 이미지의 특징을 압축한 뒤,  
조향값을 함께 사용해서 다음 상태를 예측하고,  
그 결과를 다시 이미지로 복원하여 **다음 프레임**을 만들어냅니다.

즉, 단순히 이미지를 분류하는 것이 아니라  
**현재 상황과 차량의 행동을 보고 다음 장면이 어떻게 바뀔지 예측하는 작은 형태의 world model**을 구현한 것입니다.

## 2026-03-18 1차 결과

학습 결과, 도로와 하늘, 배경 같은 **전체적인 장면 구조는 어느 정도 비슷하게 예측**하지만, 대체적으로 뿌옇게 보임
하지만 차선이나 도로의 곡선처럼 **세부적인 부분은 많이 흐리게** 나옴

이 결과를 통해, 단순한 구조만으로는 다음 프레임을 선명하게 예측하기 어렵고,  
더 좋은 결과를 위해서는 더 많은 시간 정보나 더 정교한 모델이 필요하다는 점을 확인할 수 있었음.

<img width="522" height="392" alt="epoch_010" src="https://github.com/user-attachments/assets/ab0ab27a-267e-41e1-9ec3-a0230e2954b5" />

## 2026-03-23 2차
단일 프레임만을 입력으로 사용할 경우,  
차량이 어떤 방향으로 움직이고 있는지에 대한 시간적 정보가 부족하다는 한계를 볼 수 있었다.

**개선한 것**
- 최근 4개의 연속된 프레임을 함께 입력으로 사용  
  (단일 프레임 → multi-frame 입력)

- 기존의  z_t+1을 바로 출력하지 않고 잔차를 더해 출력하는 z_t + delta_z
  완전히 변한 새로운 장면이 아닌, 조금 변한 장면 -> 현재 상태에서 얼마나 바뀌는가 방식

- 기존의 x_t → x_{t+1} 구조에서 
[x_{t-3}, x_{t-2}, x_{t-1}, x_t] + a_t → x_{t+1} 으로 변경

