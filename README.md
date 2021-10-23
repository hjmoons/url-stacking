# 악성 URL 탐지 모델

악성 URL 탐지를 위해 앙상블 기법을 사용한 딥러닝 모델. 
기존에 존재하는 CNN, LSTM, GRU 모델을 스태킹 기법에 적용하여 모델 구현 


## Model
|모델|설명|
|:---:|:---|
|<strong>CNN</strong><br><p align="center"><img src="img/cnn.png" width="300"></p>|- 1차원 시퀀스 데이터인 URL을 분류하기 위해 1DCNN을 사용<br>- `특성 추출 단계`: 성능 향상을 위해 다양한 커널 사이즈에서 특징을 추출하도록 커널 사이즈가 서로 다른 4개의 1D Convolution 연산을 사용<br>- `분류 단계`: 특성 추출 단계의 결과를 concatenation하고 hidden layer를 통해 분류 결과를 출력|
|<strong>LSTM</strong><br><p align="center"><img src="img/lstm.png" width="300"></p>|- 시퀀스 데이터인 URL을 높은 성능으로 분류하기 위해 사용<br>- LSTM Layer, Dropout, Fully-Connected Layer로 구성<br>- LSTM Layer 출력 차원: 128, Dropout 비율: 0.5|
|<strong>GRU</strong><br><p align="center"><img src="img/gru.png" width="300"></p>|- GRU 모델은 LSTM보다 계산량을 줄인 모델<br>- GRU Layer, Dropout, Fully-Connected Layer로 구성<br>- GRU Layer 출력 차원: 128, Dropout 비율: 0.5|
|<strong>FINAL</strong><br><p align="center"><img src="img/stacking.png" width="400"></p>|- 스태킹(Stacking): 여러 베이스 모델에서 예측하여 나온 결과 값을 스태킹 모델에 입력으로 하여 다시 모델을 학습시키는 기법<br>- 베이스 모델: CNN, LSTM, GRU 모델 사용<br>- 베이스 모델에서 예측한 결과 값을 입력으로 하여 Fully-Connected Layer로 구성된 스태킹 모델 학습|

## Accuracy

<table>
    <td>Data</td>
    <td>Model</td>
    <td>Accuracy</td>
    <tr>
        <td rowspan="4">URL</td>
        <td>CNN</td>
        <td>0.9355</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>0.9685</td>
    </tr>
    <tr>
        <td>GRU</td>
        <td>0.96</td>
    </tr>
    <tr>
        <td>Stacking</td>
        <td>0.9715</td>
    </tr>
    <tr>
        <td rowspan="4">DGA</td>
        <td>CNN</td>
        <td>0.9337</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>0.9552</td>
    </tr>
    <tr>
        <td>GRU</td>
        <td>0.9501</td>
    </tr>
    <tr>
        <td>Stacking</td>
        <td>0.959</td>
    </tr>
</table>

## Version
```
tensorflow 1.13.1
keras 2.2.4
```