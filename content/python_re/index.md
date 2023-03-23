---
emoji: ✒️
title: Re 정규화 라이브러리 사용법
date: '2022-06-01 23:00:00'
author: 하성민
tags: blog gatsby theme 개츠비 테마
categories: python
---



# Re

## 1. 기초 문법

[ ]
이 안에있는건 그대로 들어간다

[a-zA-Z] 알파벳 전부  
[0-10] 숫자 전부  
[.?,!] 이건그냥 쓴거 그대로를 의미한다 

---
a.b 여기서 .은 모든 문자를 의미함
대신 문자가 하나라도 들어는 가야함

---
a* b 이거는 바로 앞 문자 a 가 무한대로 반복된다는 의미  
이건 a가 0번 반복되는것도 포함이다  

ca* b 이런것도 a 만 포함하는거다 c는 제외다.

---
ca+t 이거는 최소 1번 이상은 반복해야 한다 

---
{} 이거는 반복횟수 고정  
a{m,n} 이거는 a가 m~n번만 반복해야 한다.

그런의미에서 a* 는 a{0,} 와 같고  
a+ 는 a{1,} 와 같다  

특정 반복수 n 은 a{n} 으로 하면 된다

---
? 는 {0,1} 을 의미한다.   
ca?t 은 a가 없거나 1번만 있으면 된다.  

---
^ 문자열의 처음  , ^python : 무조건 시작이 python 이어야 함  
 문자열의 마지막  python $
 
 
 ---
 \d 숫자와 매치  \D 반대  
 \s 공백과 매치 [ \t\n\r\f\v] 와 동일 \S 반대  
 \w 문자+숫자 매치 [^a-zA-Z0-9_] 와 동일 \W 반대  
근데 이걸 쓸라면 앞에 r을 붙여야 하는 듯



r'' : 그 안의 것이 위의 저 약속이 아닌  raw데이터임을 나타내주는 문법

## 2. 사용법


```python
import re

p = re.compile('[a-z]+')
```

compile 메서드를 사용해 정규 표현식을 컴파일.

컴파일된 re 객체는 다음과 같은 메서드 사용 가능

match() : 처음부터 순서대로 매치 (match객체)  (뒤에 있어도 처음에 없으면 막힘)  
search() : 문자열 전체 매치 (match객체)    
findall() : 매치되는 모든 문자열을 list로 반환    
finiter() : 매치되는 모든 문자열을 반복가능한 match 개체로 반환    

---
group() 매치된 문자열을 반환  
start() 매치된 문자열의 시작 위치를 반환  
end() 매치된 문자열의 끝 위치 반환  
span() 매치된 문자열의 (시작,끝) 을 튜플로 반환  

예시)


```python
m = p.search("3 python")
```


```python
m
```




    <re.Match object; span=(2, 8), match='python'>




```python
m.group()
```




    'python'




```python
m.start()
```




    2




```python
m.span()
```




    (2, 8)



이걸 한 줄로 할 수 있다.


```python
m = re.match('[a-z]+','3 python')
```

앞에 컴파일할 정규표현식, 그리고 뒤에 검색할 데이터를 넣으면 된다.
