---
emoji: 😁
title: 인공지능 기초 레이어 이해하기
date: '2022-04-21 23:13:00'
author: 하성민
tags: blog gatsby theme 개츠비 테마
categories: STUDY
---

# <span style='background-color: #fff5b1'>딥러닝 네트워크를 구성하는 레이어, 이게 뭘까?</span>
ANN : artificial NN 인공신경망

딥러닝은 y = Wx + by=Wx+b 에서 최적의 `레이어 W(Weight)`과 `편향 b`를 찾는 과정!

레이어 로 이루어져 있는데.

레이어 : 하나의 물체가 여러개의 논리적인 세부 객체로 구성되어 있는 경우, 그 내부 객체를 이르는 말

Linear

Convolutional

Embedding

Recurrent 

이렇게 있음 레이어들이.

Fully Connected Layer,  
Feedforward Neural Network,  
Multilayer Perceptrons,  
Dense Layer... 

등 다양한 이름으로 불리지만 그 모든 것들은 결국 Linear 레이어에 해당

선형 대수학의 선형변환 (Linear Transform) 과 동일한 기능을 한다.




```python
import tensorflow as tf

batch_size = 64
boxes = tf.zeros((batch_size, 4, 2))     # Tensorflow는 Batch를 기반으로 동작하기에,
                                         # 우리는 사각형 2개 세트를 batch_size개만큼
                                         # 만든 후 처리를 하게 됩니다.
print("1단계 연산 준비:", boxes.shape)

first_linear = tf.keras.layers.Dense(units=1, use_bias=False) 
# units은 출력 차원 수를 의미합니다.
# Weight 행렬 속 실수를 인간의 뇌 속 하나의 뉴런 '유닛' 취급을 하는 거죠!

first_out = first_linear(boxes)
first_out = tf.squeeze(first_out, axis=-1) # (4, 1)을 (4,)로 변환해줍니다.
                                           # (불필요한 차원 축소)

print("1단계 연산 결과:", first_out.shape)
print("1단계 Linear Layer의 Weight 형태:", first_linear.weights[0].shape)

print("\n2단계 연산 준비:", first_out.shape)

second_linear = tf.keras.layers.Dense(units=1, use_bias=False)
second_out = second_linear(first_out)
second_out = tf.squeeze(second_out, axis=-1)

print("2단계 연산 결과:", second_out.shape)
print("2단계 Linear Layer의 Weight 형태:", second_linear.weights[0].shape)
```

    1단계 연산 준비: (64, 4, 2)
    1단계 연산 결과: (64, 4)
    1단계 Linear Layer의 Weight 형태: (2, 1)
    
    2단계 연산 준비: (64, 4)
    2단계 연산 결과: (64,)
    2단계 Linear Layer의 Weight 형태: (4, 1)


축소만 하는 방식


```python
import tensorflow as tf

batch_size = 64
boxes = tf.zeros((batch_size, 4, 2))

print("1단계 연산 준비:", boxes.shape)

first_linear = tf.keras.layers.Dense(units=3, use_bias=False)
first_out = first_linear(boxes)

print("1단계 연산 결과:", first_out.shape)
print("1단계 Linear Layer의 Weight 형태:", first_linear.weights[0].shape)

print("\n2단계 연산 준비:", first_out.shape)

second_linear = tf.keras.layers.Dense(units=1, use_bias=False)
second_out = second_linear(first_out)
second_out = tf.squeeze(second_out, axis=-1)

print("2단계 연산 결과:", second_out.shape)
print("2단계 Linear Layer의 Weight 형태:", second_linear.weights[0].shape)

print("\n3단계 연산 준비:", second_out.shape)

third_linear = tf.keras.layers.Dense(units=1, use_bias=False)
third_out = third_linear(second_out)
third_out = tf.squeeze(third_out, axis=-1)

print("3단계 연산 결과:", third_out.shape)
print("3단계 Linear Layer의 Weight 형태:", third_linear.weights[0].shape)

total_params = \
first_linear.count_params() + \
second_linear.count_params() + \
third_linear.count_params()

print("총 Parameters:", total_params)
```

    1단계 연산 준비: (64, 4, 2)
    1단계 연산 결과: (64, 4, 3)
    1단계 Linear Layer의 Weight 형태: (2, 3)
    
    2단계 연산 준비: (64, 4, 3)
    2단계 연산 결과: (64, 4)
    2단계 Linear Layer의 Weight 형태: (3, 1)
    
    3단계 연산 준비: (64, 4)
    3단계 연산 결과: (64,)
    3단계 Linear Layer의 Weight 형태: (4, 1)
    총 Parameters: 13


한번 증가(2,3) 시켰다가 축소시키는 방식


파라미터를 늘리면 (2,3) (3,1) (4,1) = 13  `(use_bias=False)` 일때
더 많은 데이터를 보존할 수는 잇겟지만

`use_bias` 하면 3 + 1 + 1 해서 총 18 파라미터가 된다


너무 많은 파라미터는 과적합을 초래한다.

### convlutional 레이어

필터(커널)를 만들어서 그 필터만큼의 픽셀값들을 다 곱한다음 더해 다음 레이어로 출력  

![image.png](attachment:image.png)

보통 3x3 사이즈 등의 커널을 만든다

커널의 이동 사이즈를 stride 라고 부른다

convolutional 레이어는 필터 개수 x필터 가로 x 필터 세로 로 이루어진 weight 값을 가진다


```python
import tensorflow as tf

batch_size = 64
pic = tf.zeros((batch_size, 1920, 1080, 3))

print("입력 이미지 데이터:", pic.shape)
conv_layer = tf.keras.layers.Conv2D(filters=16,
                                    kernel_size=(5, 5),
                                    strides=5,
                                    use_bias=False)
conv_out = conv_layer(pic)

print("\nConvolution 결과:", conv_out.shape)
print("Convolution Layer의 Parameter 수:", conv_layer.count_params())

flatten_out = tf.keras.layers.Flatten()(conv_out)
print("\n1차원으로 펼친 데이터:", flatten_out.shape)

linear_layer = tf.keras.layers.Dense(units=1, use_bias=False)
linear_out = linear_layer(flatten_out)

print("\nLinear 결과:", linear_out.shape)
print("Linear Layer의 Parameter 수:", linear_layer.count_params())
```

    입력 이미지 데이터: (64, 1920, 1080, 3)
    
    Convolution 결과: (64, 384, 216, 16)
    Convolution Layer의 Parameter 수: 1200
    
    1차원으로 펼친 데이터: (64, 1327104)
    
    Linear 결과: (64, 1)
    Linear Layer의 Parameter 수: 1327104


## pooling layer

컨볼류셔널 레이어는 필터 사이즈에 의존하게 된다.

근데 필터사이즈를 키우다보면 결국 컨볼루셔널 레이어의 정체성이 약해지는데  

그래서 필터사이즈가 아닌 receptive Field (수용 영역)을 키워야 한다.

수용 영역:  출력 레이어의 뉴런 하나에 영향을 미치는 입력 뉴런들의 공간 크기
(그럼 컨볼류셔널에서는 커널사이즈와 같다.)

맥스풀링도 맥스풀링 사이즈가 나와야 되는 거 아닌가?

맥스풀링을 하면 수용영역의 크기는 키울 수 잇지만,  
파라미터 사이즈는 늘지 않는다.

장점

1. translate invariance 

2. Non-linear 함수와 동일한 특징 추출 효과

3. 수용 영역 (receptive Field) 극대화 효과

```toc

```

