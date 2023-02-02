---
emoji: 😁
title: pre-trained 모델 가져오는 링크
date: '2022-04-21 23:11:00'
author: 하성민
tags: blog gatsby theme 개츠비 테마
categories: DeepML
---

# <span style='background-color: #fff5b1'>미리 학습된 딥러닝 Pre-trained  Deep Learning 사용처</span>
요즘 핫한 만큼 다양한 연구와 기법이 발전되고 있다

DNN
딥 뉴럴 네트워크

더 좋은 딥 네트워크를 만들기 위해 많은 종류의 네트워크가 생겼다.  
그 중 몇 가지 사전학습된 pre-trained Network 는
TF 나 Pytorch 등의 프레임워크 차원으로 지원하고 있다.

---
우리는 그 많은 모델들을 훑어볼 건데  

특히 ResNet 과 VGG 를 중심적으로 볼 거다

## Image Net

2010년 ILSVRC 2010 을 시작으로 대량의 이미지 데이터셋  
만 개가 넘는 카테고리에 100만 장 규모의 사진을 가지고 있다.  

이걸 통해 많은 사람들이 이미지 분류 콘테스트에 나가 네트워크를 형성했다.

1. AlexNet
2011년 이미지넷 챌린지 1등 모델. 논문저자의 이름을 땄다.  
CNN 구조의 확장판이다.  
2개의 GPU 로 병렬연산을 수행하기 위해 병렬구조로 설계되었다.  


[자세한 내용](https://bskyvision.com/421)

* LeNet 
이건 이때 생긴건 아니지만 1998년에 개발한 CNN 알고리즘 이름이다.

LeNet-5는 인풋, 3개의 컨볼루션 레이어(C1, C3, C5), 2개의 서브샘플링 레이어(S2, S4), 1층의 full-connected 레이어(F6), 아웃풋 레이어로 구성되어 있다. 참고로 C1부터 F6까지 활성화 함수로 tanh을 사용한다. 


2. VGG (VGG16, VGG19 등)
2014년 이미지넷 챌린지 준우승 모델  
이름처럼 16, 19개의 층을 이룸.  
병렬구조가 아니다. 

---

근데 추세를 보면 계속 층이 깊어지는게  
좋다고 하는데,  이게 또

막 층을키운다고만 좋은게 아니다.

부작용이 있다.
- vanishing gradient (또는 Exoloding Gradient)

와 근데 이걸 해결한 것이  

3. ResNet
2015년 이미지넷 챌린지 우승 모델
Skip connection 이라는 구조로 해결
: 레이어의 입력을 다른 곳에 이어서 Gradient 가 깊게 이어지도록 만드는 구조


---

## 이제는 실습으로 만들어보자

https://github.com/keras-team/keras-applications/tree/master/keras_applications

그냥 여기에 다 담겨있다고 보면 된다

keras 에서 지원하는 pre-trained model 이 담겨있다.
굿굿 킹왕짱 굿굿 🚶‍♂️🧓👩👨


```toc

```


```python

```
