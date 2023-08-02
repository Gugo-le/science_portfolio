# Chat GPT와 같은 인공지능은 어떻게 우리 말을 이해하는 걸까?

## 탐구 영역


인공지능, 컴퓨터 과학


## 탐구 동기 및 목적

사실 chat gpt 뿐만 아니라 음성 비서, 번역기 등, 기계들이 우리 말을 이해하고 OUTPUT을 내는 것이 평소 큰 궁금증이었다. 어떠한 방식으로 이해하는지 탐구하고 실제 모델을 직접 만들어 탐구해보자!

---
## 탐구 일정

## 탐구 내용

1. 자연어 처리 소개
2. 텍스트 처리
3. 언어 모델
4. 유사도 분석

### 1. 자연어 처리 소개

- 음성인식 Speech Recognition
- 번역 Translation
- 요약 Text Summary
- 분류 Text Classification

---

## Chatbot: A Program for Interaction

```
Sentiment Analysis
텍스트에 녹아 있는 감성 또는 의견을 파악

Tokenization
단어의 최소한의 의미를 파악하는 쪼개기

Named Entity Recognition
텍스트로부터 주제 파악하기

Noemalization
의도된 오타 파악하기

Dependency Parsing
문장 구성 성분의 분석
```

## SIRI: An assistant for Questions

```
Feature Analysis
음성 데이터로부터 특징을 추출

Language Model
언어별로 갖고 있는 특성을 반영

Deep Learning
이미 학습된 데이터로부터 음성 신호 처리

HMM: Hiddem Markov Model
앞으로 나올 단어 또는 주제의 예측

Similarity Analysis
음성 신호가 어떤 기준에 부합하는가?
```

## Translator: An assistant for Questions

```
Encoding
유사도 기반 자연어의 특징 추출

Time Series Modeling
문장을 시간에 따른 데이터로 처리

Attention Mechanism
번역에 필요한 부분에만 집중하기

Self-Attention
문장 사이의 상관관계를 분석하기

Transformer
Attention 구조를 이용한 번역 원리
```

### 2. 텍스트 전처리 과정

- 토큰화(단어 기준으로 자른다.)
- 정제 및 추출(중요한 단어만!)
- 인코딩(남겨진 단어들을 정수 또는 벡터로 바꾼다.)

## 언어의 형태소

화분에 예쁜 꽃이 피었다.<br>
      ↓<br>
화분(명사) + 에(조사) + 예쁘(어간) + ㄴ(어미) + 꽃 (명사) + 이(조사) + 피(어간) + 었(어미) + 다(어미)
- 자립형태소: 명사, 수사, 부사 , 감탄사
- 의존 형태소: 조사, 어미, 어간

## 모른다 = ??

> 모르네
> 모르데
> 모르지
> 모르더라
> 모르니
> 모르고
> 모르면
> 모르겠다면
> 모르겠는데
> 모르겠지만
> 몰랐지만
> 몰랐겠어
> .
> .
> .

다 다른 의미로 받아들인다

### 언어 전처리

: 컴퓨터 및 컴퓨터 언어에서 자연어를 효과적으로 처리할 수 있도록 "전처리" 과정을 거친다.

<과정><br>
> Sentence -> Tokenization -> Cleaning, Stemming -> Encoding -> Sorting -> Padding, Similarity

## 토큰화(Tokenization)
- 표준 토큰화: Treebank Tokenization

``` python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
text = "They ignore our need to obtain a deep understanding of a subject, which includes memorizing and storing a richly structured database"   
print(tokenizer.tokenize(text))

```

```
['They', 'ignore', 'our', 'need', 'to', 'obtain', 'a', 'deep', 'understanding', 'of', 'a', 'subject', ',', 'which', 'includes', 'memorizing', 'and', 'storing', 'a', 'richly', 'structured', 'database']
```

## 정제 및 추출

- 정제: 데이터 사용 목적에 맞추어 노이즈를 제거

1: 대문자 vs 소문자<br>
> 서로 다르게 본다.

2: 츌현 횟수가 적은 단어의 제거
>
3: 길이가 짭은 단어, 지시(대)명사, 관사의 제거

- 추출 
> 어간(Stem): 단어의 의미를 담은 핵심

> 접사(Affix): 단어에 추가 용법을 부여

- 어간 추출: 춤사 정보를 갖고 있지 않음
- 표제어 추출: 품사 정보를 포함하고 있음

## 불용어(Stopword)
: 문장에서 대체로 작용하지 않는, 중요도가 낮은 단어 제거
> 조사, 접미사 같은 단어들은 문장에서는 자주 등장하지만 실제 의미 분석을 하는데는 거의 기여하는 바가 없는 경우

## 정규 표현식(Regular Expression)
: 특정 규칙이 있는 텍스트 데이터를 빠르게 정제할 수 있다.

## 원-핫 인코딩
: 컴퓨터 또는 기계는 문자보다는 숫자를 더 잘 처리 할 수 있는데 이를 위해 자연어 처리에서는 문자를 숫자로 바꿀 수 있다.
> 단어 집합의 크기를 벡터의 파원으로 하고, 표현하고 싶은 단어의 인덱스의 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다. (->원-핫 벡터)