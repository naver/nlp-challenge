### Q. 특별한 개발 지식이나 언어 사용 경험이 없어도 참여 가능한가요?
기반 지식은 필요하며, AI 지식, 특히 NLP 관련 문제를 해결 해 본 경험이 있다면 더 좋은 결과를 낼 수 있습니다.

### Q. 참가 정보를 변경하고 싶어요!
변경할 정보를 dl_nlp_challenge@navercorp.com으로 보내주세요.

### Q. 미션은 둘 중 하나만 도전할 수 있나요?
하나만 해도 되고 둘 다 해도 됩니다. 시상은 미션별로 할 예정입니다.

### Q. 대회는 온라인으로만 진행하나요?
네. 온라인으로만 진행합니다.

### Q. 언제 데이터가 오픈되나요?
훈련 데이터는 11월 16일에 오픈 될 예정입니다.

### Q. 공개된 데이터 이외에 추가 데이터를 사용해도 될까요?
가능합니다. 예를 들어 pre-train된 embedding model을 활용한다든지, 한국어 wikipedia 사전이나 다른 사전등을 활용하셔도 됩니다.
단, 다음 조건을 만족해야 합니다.
  - 외부에 공개된 데이터이며, 누구나 접근 가능한 데이터여야 한다.
  - 사용된 외부 데이터의 종류 및 접근 방법(원소스)는 추후 공개되어야 한다.
  - 사용방법은 다음과 같습니다.
     1. 코드상에서 특정위치에 data/pretrained model을 다운받고 training/ model load 하는 부분을 임의로 만드셔서 생성된 모델을 ```nsml.save(model name)``` 함수를 사용해서 모델을 저장하시면 됩니다. (코드 내에서 다 작성 하시면 됩니다.)
     2. NSML 에서는 코드를 실행할때 도커환경에서 실행 되는데요, 원하는 data, pretrained model 이 저장된 docker image를 직접 빌드하셔서 docker hub에 푸시해서 사용하시거나, 빌드되어있는 docker image 이름을 NSML 의 setup.py, requirements.txt 에 기입하여서 사용하는 방법이 있습니다.[참조](https://n-clair.github.io/nlp-challenge-docs/_build/html/ko_KR/contents/session/prepare_a_session_environment.html#)

### Q. NSML에서 사용하는 언어는 뭔가요?
NSML은 네이버에서 개발한 머신러닝 클라우드 플랫폼입니다.CLI와 웹 등의 인터페이스를 통해 클라우드 자원을 사용할 수 있습니다. tensorflow, Keras, Pytorch 등을 비롯해 파이썬 기반의 머신 러닝 라이브러리는 대부분 사용할 수 있습니다.

### Q. 경진대회에서 개발한 알고리즘을 개인 논문에 사용해도 되나요?
다음 사항을 준수한다면 사용할 수 있습니다.
* 네이버 NLP Challenge에 참여해 개발한 알고리즘이라는 점을 Acknowledgement에 명시
* 데이터 출처를 명시 (명시 방법은 창원대학교에 문의 부탁드립니다 http://air.changwon.ac.kr/ )
