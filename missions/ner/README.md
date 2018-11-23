## NER Model Baseline for NSML

### NER baseline Model 구조
* Bidirectional RNN + CRF
* 어절, 음절(RNN) Concat 하여 사용

### Training Dataset

- [개체명 인식(NER)](http://air.changwon.ac.kr/?page_id=10)

### Training
```
$ nsml run -d NER

```

### Training

학습은 다음 명령행을 실행하여 시작할 수 있습니다.
```bash
$ nsml run -e main.py -d NER
```

학습 중 다음 명령행을 실행하여 로그를 확인할 수 있습니다.
```bash
$ nsml logs -f [SESSIONNAME]
```

다음 명령행을 실행해 현재 수행중인 세션을 확인할 수 있습니다.
```bash
$ nsml ps
```

### Submit

모델의 제출은 다음 명령행으로 실행할 수 있습니다.
```bash
$nsml submit [SESSIONNAME] [CHECKPOINT]
```
'CHECKPOINT'는 세션마다 기록됩니다.
다음 명령행으로 특정 세션의 checkpoint를 확인하실 수 있습니다.

```bash
$nsml model ls [SESSIONNAME]
```
다음과 같이 세션의 checkpoint들이 나타납니다.

```bash
Checkpoint    Last Modified    Elapsed    Summary                               Size
------------  ---------------  ---------  ------------------------------------  --------
0             seconds ago      2.523      step=0, train/loss=28.56278305053711  66.7 MB
1             just now         14.424     step=1, train/loss=19.05735492706299  66.84 MB
...
```

### 데이터 로더 Return 값 형식
* 각 문장별로 [ idx, ejeols, nemed_entitis ] ( idx는 1으로 시작 )
```
ex)[
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'],
     ['-(해설)', '발목을', '삐끗하거나', '칼에', '살짝', '베인', '작은', '상처의', '통증도', '2개월', '이상', '계속되면', '만성통증으로', '발전한다', '.'],
     ['-', 'ANM_B', '-', '-', '-', '-', '-', '-', '-', 'DAT_B', 'DAT_I', '-', 'TRM_B', '-', '-']
   ]
```

### 최종 정답 태깅용 출력 형식
* 각 어절별로 모든 결과를 출력해야 합니다. ( 아래 예시 참조 )
* 없는 태그는 "-" 로 표시합니다.
```
ex)
    [
     ['-', 'ANM_B', '-', '-', '-', '-', '-', '-', '-', 'DAT_B', 'DAT_I', '-', 'TRM_B', '-', '-'],
     [ ... ]
    ]
```

### infer 최종 반환 출력 형식
* 출력 형식 [ ( prob, output ), ( prob, output ), () ... ]
* 참조 : https://github.com/naver/nlp-challenge/blob/master/missions/ner/main.py#L97
```
ex)
    [
     (0.0, ['NUM_B', '-', '-', '-']),
     (0.0, ['PER_B', 'PER_I', 'CVL_B', 'NUM_B', '-', '-', '-', '-', '-', '-']),
     ( ), ( )
    ]
```