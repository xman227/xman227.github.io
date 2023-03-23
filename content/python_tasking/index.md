---
emoji: 👩‍👩‍👦‍👦
title: 파이썬 함수 병렬 처리
date: '2022-02-23 23:00:00'
author: 하성민
tags: blog gatsby theme 개츠비 테마
categories: python
---


# 빅데이터 핸들링의 필수, 병렬 컴퓨팅

## 동시성과 병렬성

동시성 : Concurrency

병렬성 : Parallelism

멀티 태스킹에는 위의 2가지 방식있다.

동시성 : 하나의 processor 가 남는 시간동안 다른 task 를 동시에 수행

다시말해, 특정 순간에는 1가지 task 를 하겠지만, 다음 task 로 넘어가는데 시간이 걸리게 되면 다른 task 를 수행하도록 한다.

병렬성 : 여러 processor 가 각자 task 를 동시에 수행

> 다시 말해, 병렬성을 가진 processor 가 동시성을 가지고 일할 수 있다.

예를 들어, 라면을 조리하려면 물을 끓여야 하는데.  
processor 는 불을 켜고 물이 끓기를 기다려야 한다.

이처럼 대기해야 하는 상황을 'bound' 상태 라고 한다. 

bound 상태에 그저 대기만 하고 있는 방식을 Syncronzized, 동기 방식이라고 한다.  
bound 상태에 다른 일을 처리하는 방식을 Asynchronous 비동기 방식이라고 한다.

---

## process, thread , profiling

### Process

하나의 프로그램을 생성할 때, 운영체제는 하나의 프로세스 를 생성한다.

프로세스는 프로그램을 작동시키면서 일어나는 메모리상의 작업 단위를 의미한다.

하나의 프로세스는 CPU , 메모리(Ram), 디스크 및 자료구조를 이용하는데,
그 과정에서 메모리는 여러 번 쓰이게 된다.

### Thread

thread 는 process 내부에 있는 또 각각의 작업단위를 의미한다

헬스를 하는 `program` 에서, 운동선수라는 `processor` 는 스쿼트, 레그프레스, 레그 컬 등의 `thread` 를 수행한다.

헬스로 계속 비유를 들어 보겠다.

운동선수들끼리 헬스장 자체를 공유할 수는 있지만, 하나의 렉 을 공유할 수는 없다.

> 다시말해, `thraed` 마다 전용 메모리 공간 `head` 을 가진다.


### profiling

사건수사 프로파일링. 들어본적 있을 것이다.  
그 사건의 일거수 일투족을 들여다보는 것을 의미한다.

컴퓨터 내부의 프로파일링도 똑같다.
  
프로그램 코드 내부에서 
1. 어느 부분이 느린지
2. 어디서 RAM을 많이 사용하는지

확인할 수 있다.

파이썬에서도 구현할 수 있다.



```python
import timeit # 프로파일링 라이브러리
        
def f1():
    s = set(range(100))

    
def f2():
    l = list(range(100))

    
def f3():
    t = tuple(range(100))


def f4():
    s = str(range(100))

    
def f5():
    s = set()
    for i in range(100):
        s.add(i)

def f6():
    l = []
    for i in range(100):
        l.append(i)
    
def f7():
    s_comp = {i for i in range(100)}

    
def f8():
    l_comp = [i for i in range(100)]
    

t1 = timeit.Timer("f1()", "from __main__ import f1")
t2 = timeit.Timer("f2()", "from __main__ import f2")
t3 = timeit.Timer("f3()", "from __main__ import f3")
t4 = timeit.Timer("f4()", "from __main__ import f4")
t5 = timeit.Timer("f5()", "from __main__ import f5")
t6 = timeit.Timer("f6()", "from __main__ import f6")
t7 = timeit.Timer("f7()", "from __main__ import f7")
t8 = timeit.Timer("f8()", "from __main__ import f8")

print("set               :", t1.timeit())
print("list              :", t2.timeit())
print("tuple             :", t3.timeit())
print("string            :", t4.timeit())
print("set_add           :", t5.timeit())
print("list_append       :", t6.timeit())
print("set_comprehension :", t5.timeit())
print("list_comprehension:", t6.timeit())
```

    set               : 1.690346003999366
    list              : 0.7587415179996242
    tuple             : 0.7320455680001032
    string            : 0.4020238100001734
    set_add           : 5.726110922999396
    list_append       : 5.244985264999741
    set_comprehension : 5.7903866610004116
    list_comprehension: 5.160655052999573


0부터 100까지의 수를 각 자료구조에 담는 시간을 세어 보았다.

이렇게 함수의 성능을 측정할 수 있다.
다만 이건 프로파일링을 최 단순화 한 것이다.

원래는 코드별 들어가는 Ram 의 크기나 성능까지 측정한다.

## Multi-tasking

단순히 말해 우리는 병렬화, 또는 동시화 하기 위해서 자원을 up 할 수도 있고,  
자원을 out (확장) 할 수도 있다.

Scale up - 한 대의 컴퓨터 성능 향상
Scale out - 한 대에서 여러대로 컴퓨터 개수를 늘려 한대처럼 사용

## 실습




### thread


```python
from threading import *
from time import sleep

Stopped = False

def worker(work, sleep_sec): 
    
    while not Stopped:          
        print('do ', work)      
        sleep(sleep_sec) 
        
    print('retired..')         
        
        
t = Thread(target=worker, args=('work', 2))    
t.start()    
```

    do  work
    do  work
    do  work


원래는 하나의 코드가 있으면 작동중에 다른 것에 영향을 받지 않는다.  
다만 현재 `Thread` 메서드를 통해 함수를 시작했다.

`Thread` 에서 worker 메서드를 시작시켰고, 함수의 인자는 work, 2 가 들어갔다.

이 작업이 끝나려면 `Stopped = True` 가 되어야 한다. 


```python
Stopped = True    

t.join()          
print('finish')
```

    retired..
    finish


이와 같이 `join` 메서드를 통해 같은 `thread` 에서 작업을 진행시켰고
상단의 무한루프를 중단 시킬 수가 있다.

### multi - process


```python
import multiprocessing as mp

p = mp.Process(target='들어갈 프로그램/함수', args=('함수의 인자'))

p.start() # 프로세스 시작
p.join() # 실제 종료까지 기다림 (필요시에만 사용)
p.terminate() # 프로세스 종료
```


```python

def fitness():
    print('exercising...')

p = mp.Process(target=fitness, args=())
p.start()
```

    delivering...


멀티 프로세스도 `thread` 와 동일하게 구현가능하다

---
## 실제 코딩 시 적용방법

결국 우리가 이런 걸 하는 이유는 더 효율적이고 빠른 프로그램을 만들 수 있기 때문이다.

그런데 모든 함수마다 이렇게 병렬처리를 해주는 게 더 귀찮고 시간이 많이든다.

때문에 파이썬에서는 이 병렬처리 및 multi-tasking 을 자동으로 해주는 모듈이 있다.

`concurrent.futures` 이것이다.

[공식문서](https://docs.python.org/ko/3.7/library/concurrent.futures.html)

만약 병렬로 처리하고 싶은 코드가 있다면 아래와 같은 코드에

병렬을 원하는 함수를 `func` 에 넣어 사용하면 된다.


```python
import concurrent

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        for output in executor.map( func , data ): # data 를 func 로 처리.
            print('%d is prime: %s' % (output)) #모든 sequence 데이터가 동시처리됨

```

```toc

```


```python

```
