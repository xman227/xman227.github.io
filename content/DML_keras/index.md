---
emoji: 🧙‍♂️
title: 파이썬으로 딥러닝 모델 제작하기
date: '2022-02-19 23:00:00'
author: 하성민
tags: blog gatsby theme 개츠비 테마
categories: STUDY
---



## 1. 🌠 Depp Learning 모델 생성 방법 세가지


Tensor flow V2 버전에서 딥러닝 모델 작성 방법에는 크게 3가지가 있다.

- Sequential
- Functional : sequential 의 일반화된 개념
- Model Subclassing : 클래스로 구현된 기존 모델을 상속받아 자기 모델 만들기

순차적으로 어떤 차이가 있고,  
어떤 식으로 제작하는지 알아가보자.

---



### 1. Sequential Model  


```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(__넣고싶은 레이어__)
model.add(__넣고싶은 레이어__)
model.add(__넣고싶은 레이어__)

model.fit(x, y, epochs=20, batch_size=16)
```

epochs : 모델로 데이터를 학습할 횟수  
batch_size : 데이터를 소분해 넣을 양

model = keras.Sequential() 을 활용하면  
딥러닝 모델을 쌓아 나갈 수 있다.

입력부터 출력까지 순차적(시퀀셜) 으로 add 하면 된다.

but, 모델의 입력과 출력이 여러개인 경우에는 적합하지 않다.  
(반드시 입력 1개 출력 1가지 여야 함)

### 2. Functional API


```python
import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(__원하는 입력값 모양__))
x = keras.layers.__넣고싶은 레이어__(관련 파라미터)(input)
x = keras.layers.__넣고싶은 레이어__(관련 파라미터)(x)
outputs = keras.layers.__넣고싶은 레이어__(관련 파라미터)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.fit(x,y, epochs=10, batch_size=32)
```

model 에 ketas.Model 이 들어간다.  
이것은 우리가 danse 나 Flatten 같은 짜여져 있는 신경망을 쓰는 게 아니라  
직접 input 과 output 을 구성한다.

때문에 입력과 출력값이 자유롭다

---
딥 러닝 모델은 일반적으로 레이어의 DAG Directed Acyclic graph 이다.  
레이어의 그래프를 bulid 한다는 뜻이다.


### 3. Subclassing


```python
import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.__정의하고자 하는 레이어__()
        self.__정의하고자 하는 레이어__()
        self.__정의하고자 하는 레이어__()
    
    def call(self, x):
        x = self.__정의하고자 하는 레이어__(x)
        x = self.__정의하고자 하는 레이어__(x)
        x = self.__정의하고자 하는 레이어__(x)
        
        return x
    
model = CustomModel()
model.fit(x,y, epochs=10, batch_size=32)
```

제일 자유로운 모델링이 가능한 subclassing  
이를 바탕으로 직접 구현해보자


```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
```


```python
# 데이터 구성부분
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=x_train[...,np.newaxis]
x_test=x_test[...,np.newaxis]

print(len(x_train), len(x_test))
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    11501568/11490434 [==============================] - 0s 0us/step
    60000 10000


## 2. 실제 구현
### 1. Sequential model


```python
# Sequential Model을 구성해주세요.
"""
Spec:
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
                           ])


"""
model.add.Conv2D(32, 3, activation="relu")
model.add.Conv2D(64, 3, activation="relu")
model.add.Flatten()
model.add.Dense(128, activation='relu')
model.add.Dense(10, activation='softmax')
"""
```




    '\nmodel.add.Conv2D(32, 3, activation="relu")\nmodel.add.Conv2D(64, 3, activation="relu")\nmodel.add.Flatten()\nmodel.add.Dense(128, activation=\'relu\')\nmodel.add.Dense(10, activation=\'softmax\')\n'




```python
# 모델 학습 설정

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

    Epoch 1/5
    1875/1875 [==============================] - 31s 3ms/step - loss: 0.1118 - accuracy: 0.9654
    Epoch 2/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0368 - accuracy: 0.9882
    Epoch 3/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0211 - accuracy: 0.9933
    Epoch 4/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0131 - accuracy: 0.9956
    Epoch 5/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0100 - accuracy: 0.9968
    313/313 - 1s - loss: 0.0516 - accuracy: 0.9878





    [0.051644984632730484, 0.9878000020980835]



### 2. Functional API


```python
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=x_train[...,np.newaxis]
x_test=x_test[...,np.newaxis]

print(len(x_train), len(x_test))
```

    60000 10000



```python
"""
Spec:
0. (28X28X1) 차원으로 정의된 Input
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

inputs = keras.Input(shape=(28,28,1))
x = keras.layers.Conv2D(32,3,activation='relu')(inputs)
x = keras.layers.Conv2D(64,3,activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

    Epoch 1/5
    1875/1875 [==============================] - 7s 3ms/step - loss: 0.1039 - accuracy: 0.9679
    Epoch 2/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0339 - accuracy: 0.9895
    Epoch 3/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0204 - accuracy: 0.9931
    Epoch 4/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0130 - accuracy: 0.9959
    Epoch 5/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0093 - accuracy: 0.9972
    313/313 - 1s - loss: 0.0522 - accuracy: 0.9868





    [0.05223735421895981, 0.9868000149726868]



### 3. Subclassing API

keras.models 를 상속받는 클래스를 만드는 것

1. __init__ 메서드에 레이어 선언
2. call() 메서드에 forward propagation 방식 체계 구현


```python
"""
Spec:
0. keras.Model 을 상속받았으며, __init__()와 call() 메서드를 가진 모델 클래스
1. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. 64개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
3. Flatten 레이어
4. 128개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
5. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
6. call의 입력값이 모델의 Input, call의 리턴값이 모델의 Output
"""


class kerasModel(keras.Model):
    def __init__(self):
        super().__init__()
        
        self.Conv2D32 = keras.layers.Conv2D(32, 3, activation='relu')
        self.Conv2D64 = keras.layers.Conv2D(64, 3, activation='relu')
        self.Flatten = keras.layers.Flatten()
        self.Dense128 = keras.layers.Dense(128, activation='relu')
        self.Dense = keras.layers.Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.Conv2D32(x)
        x = self.Conv2D64(x)
        x = self.Flatten(x)
        x = self.Dense128(x)
        output = self.Dense(x)
        
        return output
    
model = kerasModel()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

```

    Epoch 1/5
    1875/1875 [==============================] - 7s 3ms/step - loss: 0.1011 - accuracy: 0.9689
    Epoch 2/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0339 - accuracy: 0.9893
    Epoch 3/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0184 - accuracy: 0.9938
    Epoch 4/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0136 - accuracy: 0.9954
    Epoch 5/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.0078 - accuracy: 0.9973
    313/313 - 1s - loss: 0.0581 - accuracy: 0.9853





    [0.05812956392765045, 0.9853000044822693]



이건 결론적으로는 model 을 갖다 쓰는거라서  
input 을 따로 설정안하고  
그리고 call 메서드도 fit 할때 자동으로 써지는 듯 하다.


---
## 3. CIFAR -100 데이터 예제 사용하기

### 1. Sequential


```python
# 데이터 구성부분
cifar100 = keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(len(x_train), len(x_test))
```

    50000 10000



```python
# Sequential Model을 구성해주세요.
"""
Spec:
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,3,activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(100,activation='softmax')
])

```


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
```

    Epoch 1/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 2/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 3/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 4/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 5/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    313/313 - 1s - loss: nan - accuracy: 0.0100





    [nan, 0.009999999776482582]



### 2. Functional


```python
"""
Spec:
0. (32X32X3) 차원으로 정의된 Input
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
"""

inputs = tf.keras.Input(shape=(32,32,3))
x = tf.keras.layers.Conv2D(16,3,activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D((2,2))(x)
x = tf.keras.layers.Conv2D(32,3,activation='relu')(x)
x = tf.keras.layers.MaxPool2D((2,2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256,activation='relu')(x)
outputs = tf.keras.layers.Dense(100,activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5)

model.evaluate(x_test, y_test, verbose =2)
```

    Epoch 1/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 2/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 3/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 4/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 5/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    313/313 - 1s - loss: nan - accuracy: 0.0100





    [nan, 0.009999999776482582]



### 3. Subclass API


```python
"""
Spec:
0. keras.Model 을 상속받았으며, __init__()와 call() 메서드를 가진 모델 클래스
1. 16개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
2. pool_size가 2인 MaxPool 레이어
3. 32개의 채널을 가지고, 커널의 크기가 3, activation function이 relu인 Conv2D 레이어
4. pool_size가 2인 MaxPool 레이어
5. 256개의 아웃풋 노드를 가지고, activation function이 relu인 Fully-Connected Layer(Dense)
6. 데이터셋의 클래스 개수에 맞는 아웃풋 노드를 가지고, activation function이 softmax인 Fully-Connected Layer(Dense)
7. call의 입력값이 모델의 Input, call의 리턴값이 모델의 Output
"""


class kerasModel(keras.Model):
    def __init__(self):
        super().__init__()
        
        self.Conv2D16 = keras.layers.Conv2D(16, 3, activation='relu')
        self.Conv2D32 = keras.layers.Conv2D(32, 3, activation='relu')
        self.Maxpool = keras.layers.MaxPool2D((2,2))
        self.Flatten = keras.layers.Flatten()
        self.Dense256 = keras.layers.Dense(256, activation='relu')
        self.Dense = keras.layers.Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.Conv2D16(x)
        x = self.Maxpool(x)
        x = self.Conv2D32(x)
        x = self.Maxpool(x)
        x = self.Flatten(x)
        x = self.Dense256(x)
        output = self.Dense(x)
        
        return output
    
model = kerasModel()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

```

    Epoch 1/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 2/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 3/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 4/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    Epoch 5/5
    1563/1563 [==============================] - 5s 3ms/step - loss: nan - accuracy: 0.0100
    313/313 - 1s - loss: nan - accuracy: 0.0100





    [nan, 0.009999999776482582]



---

우리는 model.compile 에 있어서 계속 동일한 구성을 유지했다.

잠시 인공지능 학습 과정을 복기해보자.
1. Forward Propagation
2. Loss 값 계산
3. 중간 레이어 값 및 loss 를 활용한 chain rule 방식의 Back propagation
4. parameter update
5. repeat

이런 과정이 tf v2 에서는 fit 에 다 담겨있다.

tf.gredient tape 는 propagation 동안 진행되는 모든 연산의 중간 레이어 값을  
tape 에 기록한다.

이를 이용해 gredient 를 계산하고 tape 를 폐기한다.

우리는 이 tape 값을 이용해 고급기법을 이용할 수 있다.


```python
import tensorflow as tf
from tensorflow import keras

# 데이터 구성부분
cifar100 = keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(len(x_train), len(x_test))

# 모델 구성부분
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D((2,2))
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D((2,2))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256, activation='relu')
        self.fc2 = keras.layers.Dense(100, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

model = CustomModel()
```

    50000 10000


지금까지는


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

에서 알아서 loss 를 통해 학습 파라미터를 수정할 수 있게 해주었다.

이번에는 tape.gradient() 를 통해서  
매 스텝 마다 발생하는 gredient 를 export,  
optimizer.apply_grediens() 를 통해 발생한 gredient 가   
model.trainable_variables 를 통해 파라미터를 업데이터 하도록 한다.  


```python
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# tf.GradientTape()를 활용한 train_step
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

위의 train_step 메서드가 바로  
fit 을 통해 1번 epoch 하는 동안의 안에서 이루어지는 steps 들의 진행방식을 나타낸것.  



```python
import time
def train_model(batch_size=32):
    start = time.time()
    for epoch in range(5):
        x_batch = []
        y_batch = []
        for step, (x, y) in enumerate(zip(x_train, y_train)):
            x_batch.append(x)
            y_batch.append(y)
            if step % batch_size == batch_size-1:
                loss = train_step(np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32))
                x_batch = []
                y_batch = []
        print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
    print("It took {} seconds".format(time.time() - start))

train_model()
```

    Epoch 0: last batch loss = 3.2522
    Epoch 1: last batch loss = 2.6932
    Epoch 2: last batch loss = 2.4441
    Epoch 3: last batch loss = 2.3071
    Epoch 4: last batch loss = 2.2136
    It took 85.202951669693 seconds


위 두 함수의 연계사용이
fit 메서드를 풀어 쓴 것이다.

쉽게 말하면

데이터들을 batch_size 만큼 끊어서  
그 끊은 만큼의 x_train(feature) 를 넣어 모델 신경망에 넣고 돌려
예측한 y_train 값을 반환하고  

실제 y_trian(label) 값과 예측한 y_trian 값을 손실함수(loss) 를 돌려  
그 출력 값으로 Back propagation 을 통해 모델 내부 가중치를 새로운 가중치로 update

자 그럼 손실값은 한 1epochs 에 배치사이즈 돌린만큼 나오겠지?  
(위의 경우 32배치사이즈 이므로 데이터가 100개면 한 epochs 당 loss 가 3개 나오지)
하지만 출력되는 loss 는 맨 마지막 배치의 loss 이다.

이 과정을 epoch 횟수 만큼 반복하는 것이다.  

이게 fit 메서드가 하는 일.






---
이걸 굳이 끄집어내서 보는 이유는 

우리가 강화학습 또는 GAN 을 시행할 때 
이 내부 train_step 의 재구성을 해야 하므로!!!!!

---


```python
prediction = model.predict(x_test, batch_size=x_test.shape[0], verbose=1)
temp = sum(np.squeeze(y_test) == np.argmax(prediction, axis=1))
temp/len(y_test)  # Accuracy
```

    1/1 [==============================] - 1s 822ms/step





    0.346



일단 epoch 해봣으니 결과값도 이렇게 도출이 된다.

```toc
```