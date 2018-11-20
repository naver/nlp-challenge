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
$ nsml run -e main.py -d SRL
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