## SRL Model Baseline for NSML

### License

본 프로젝트는 최초 깃허브에 공개된 중국어 개체명 인식 프로젝트를 수정한 것입니다. ( https://github.com/zjy-ucas/ChineseNER )

### Requirement

본 프로젝트는 python3.5.2, tensorflow 1.4.1에서 테스트되었습니다.
사용 전 tensorflow-gpu 1.4.1 혹은 tensorflow 1.4.1을 설치해 주시기 바랍니다.

### Training Dataset

- [의미역 결정(SRL)](http://air.changwon.ac.kr/?page_id=14)


### Training

학습은 다음 명령행을 실행하여 시작할 수 있습니다.
```bash
$ nsml run -e main.py -d SRL
```

학습을 시작할 때 나타나는 세션의 이름을 확인해 두세요.
세션의 이름은 로그를 확인할 때, 그리고 모델을 제출할 때에 사용됩니다.
세션의 이름을 모르는 경우, 다음 명령행을 이용해 현재 실행중인 세션을 확인할 수 있습니다.

```bash
$ nsml ps
```

학습 중 다음 명령행을 실행하여 로그를 확인할 수 있습니다.
```bash
$ nsml logs -f [SESSIONNAME]
```

다음은 학습 중에 나타나는 문구 중 일부입니다.
사전의 크기, 학습 데이터셋의 크기, 각종 하이퍼 파라메터들, 그리고 매 에폭의 성능을 클래스별로 보이고 있습니다.

```bash
Found 186 unique chars
Found 11 unique tags
8 / 0 / 5 sentences in train / dev / test.
2018-10-19 22:21:45,911 - log/train.log - INFO - num_chars      :	186
2018-10-19 22:21:45,911 - log/train.log - INFO - char_dim       :	100
2018-10-19 22:21:45,911 - log/train.log - INFO - num_tags       :	11
2018-10-19 22:21:45,911 - log/train.log - INFO - seg_dim        :	0
2018-10-19 22:21:45,911 - log/train.log - INFO - char_lstm_dim  :	100
2018-10-19 22:21:45,911 - log/train.log - INFO - word_lstm_dim  :	100
2018-10-19 22:21:45,911 - log/train.log - INFO - batch_size     :	20
2018-10-19 22:21:45,912 - log/train.log - INFO - clip           :	5
2018-10-19 22:21:45,912 - log/train.log - INFO - dropout_keep   :	0.5
2018-10-19 22:21:45,912 - log/train.log - INFO - optimizer      :	adam
2018-10-19 22:21:45,912 - log/train.log - INFO - lr             :	0.001
2018-10-19 22:21:45,912 - log/train.log - INFO - pre_emb        :	False
2018-10-19 22:21:45,912 - log/train.log - INFO - lower          :	True
2018-10-19 22:21:45,912 - log/train.log - INFO - max_char_length:	8
2018-10-19 22:21:45,912 - log/train.log - INFO - max_word_length:	95
2018-10-19 22:21:49,806 - log/train.log - INFO - Created model with fresh parameters.
2018-10-19 22:21:50,234 - log/train.log - INFO - start training
2018-10-19 22:21:50,972 - log/train.log - INFO - evaluate:dev
2018-10-19 22:21:51,112 - log/train.log - INFO - processed 108 eojeols ; found: 0 arguments; correct: 0.
2018-10-19 22:21:51,112 - log/train.log - INFO - accuracy:  80.56%; precision:   0.00%; recall:   0.00%; FB1:   0.00
2018-10-19 22:21:51,112 - log/train.log - INFO - ARG0: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,113 - log/train.log - INFO - ARG1: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,113 - log/train.log - INFO - ARG3: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,113 - log/train.log - INFO - ARGM-EXT: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,113 - log/train.log - INFO - ARGM-LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,113 - log/train.log - INFO - ARGM-MNR: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,113 - log/train.log - INFO - ARGM-TMP: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,141 - log/train.log - INFO - evaluate:test
2018-10-19 22:21:51,174 - log/train.log - INFO - processed 64 eojeols ; found: 0 arguments; correct: 0.
2018-10-19 22:21:51,175 - log/train.log - INFO - accuracy:  76.56%; precision:   0.00%; recall:   0.00%; FB1:   0.00
2018-10-19 22:21:51,175 - log/train.log - INFO - ARG1: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,175 - log/train.log - INFO - ARG3: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,175 - log/train.log - INFO - ARGM-EXT: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,175 - log/train.log - INFO - ARGM-LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
2018-10-19 22:21:51,175 - log/train.log - INFO - ARGM-TMP: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
```

각종 하이퍼 파라메터들은 commandline argument로 입력할 수 있고, 주어진 파라메터들은 'config_file'에 학습 시작 시에 기록합니다.
아래는 기록된 config의 예제입니다.

```bash
{
  	"num_chars": 8000,
  	"char_dim": 100,
  	"num_tags": 13,
  	"seg_dim": 0,
  	"char_lstm_dim": 100,
  	"word_lstm_dim": 100,
  	"batch_size": 20,
  	"clip": 5,
	"patience": 5,
  	"dropout_keep": 0.5,
  	"optimizer": "adam",
  	"lr": 0.001,
  	"lower": true,
  	"max_char_length": 8,
  	"max_word_length": 95
}
```

validation-based earlystopping이 적용되어 있어, patience 만큼 연속으로 validation performance가 갱신되지 않으면 학습이 종료됩니다.

### Submit

모델을 학습하면 모델의 checkpoint가 기록됩니다.
다음 명령행으로 특정 세션의 checkpoint를 확인하실 수 있습니다.
```bash
$ nsml model ls [SESSIONNAME]
```

다음과 같이 세션의 checkpoint들이 나타납니다.

```bash
Checkpoint    Last Modified    Elapsed    Summary                                                           Size
------------  ---------------  ---------  -------------------------------------------------------------     --------
0             3 hours ago      0.701      step=0, epoch_total=40, epoch=0, train/loss=6.771212100982666     12.55 MB
1             3 hours ago      77.305     step=1, epoch_total=40, epoch=1, train/loss=5.207082271575928     12.63 MB
2             3 hours ago      80.772     step=2, epoch_total=40, epoch=2, train/loss=4.855630397796631     12.71 MB
...
```

다음 명령행으로 이 중 하나의 checkpoint를 리더보드에 제출할 수 있습니다.
```bash
$ nsml submit [SESSIONNAME] [CHECKPOINT]
```

예를 들어 7번 세션의 2번 checkpoint를 submit하고자 하는 경우 다음 명령행을 수행하시면 됩니다.
```bash
$ nsml submit [TEAM]/SRL/7 2
```

### LeaderBoard

매일 오후 8시, [링크](http://air.changwon.ac.kr/?page_id=14)에 성능이 업데이트됩니다.
