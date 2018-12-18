# 네이버, 창원대가 함께하는 NLP Challenge
네이버가 창원대와 한국어 자연어처리 기술 대회를 개최합니다.
개체명 인식(Named-entity Recognition, NER)과 의미역 결정(Semantic Role Labeling, SRL)이라는 자연어 처리의 중요한 두가지 미션을 준비했습니다.
이번 대회에서는 창원대에서 마련한 대량의 한국어 데이터와 네이버의 클라우드 머신러닝 플랫폼인 NSML를 이용하여 참여할 수 있습니다.

지금 바로 네이버, 창원대가 함께 하는 NLP Challenge에 참여해서
서로의 경험을 공유하고, 다양하고 창의적인 방법으로 문제를 해결해 보세요!

# 공지 사항
- `2018-11-16 22:36` 훈련용 데이터가 공개 되었습니다 ([링크](https://github.com/naver/nlp-challenge/issues/1))
- `2018-11-20 13:12` NER baseline 시스템이 공개되었습니다. ([링크](https://github.com/naver/nlp-challenge/tree/master/missions/ner))
- `2018-11-21 16:41` 팀명 변경 및 팀원 정보 변경은 불가능 한 점 양해 부탁드립니다.
- `2018-11-26 15:48` SRL baseline 시스템이 공개되었습니다. ([링크](https://github.com/naver/nlp-challenge/tree/master/missions/srl))
- `2018-12-03 15:01` NLP Workshop 참가신청이 시작되었습니다. ([링크](https://github.com/naver/nlp-challenge/blob/master/nlp-workshop.md))
- `2018-12-03 17:26` 참가 신청 폼이 다시 열렸습니다. 추가 참가 신청이 가능한 상태입니다.
- `2018-12-10 15:17` 참가 신청이 종료되었습니다.
- `2018-12-17 20:37` 수상팀이 결정, 공지되었습니다.

## 수상팀
### NER

수상구분 | Rank | Date | 팀명 | 구성원 | F1
-- | -- | -- | -- | -- | --
대상 | 1 | 2018.12.14 7:45 | State_Of_The_Art | 박동주 (광주과학기술원) | 90.4219
우수 | 2 | 2018.12.14 0:29 | cheap_learning | 박광현, 이영훈 (전북대학교) | 90.2417
우수 | 3 | 2018.12.14 22:46 | nlp_pln | 이신의, 박장원, 박종성<br>(연세대학교 데이터공학연구실) | 89.783
장려 | 4 | 2018.12.14 15:18 | Sogang_Alzzam | 박찬민, 박영준<br>(서강대학교 자연어처리연구실) | 88.8506
장려 | 5 | 2018.12.14 23:51 | ner_master | 조민수, 박찬희, 박진욱<br>(연세대학교 데이터공학연구실) | 88.5818
장려 | 6 | 2018.12.13 19:28 | bible | 현청천 (HELLO NMS) | 88.3348


### SRL

수상구분 | Rank | Date | 팀명 | 구성원 | F1
-- | -- | -- | -- | -- | --
대상 | 1 | 2018.12.14 22:01 | Sogang_Alzzam | 박찬민, 박영준<br>(서강대학교 자연어처리연구실) | 77.6628
우수 | 2 | 2018.12.14 17:44 | KANE_team | 함영균, 김동환, 최기선<br>(KAIST SWRC) | 76.3328
우수 | 3 | 2018.12.10 19:53 | cheap_learning | 박광현, 이영훈 (전북대학교) | 76.2543
장려 | 4 | 2018.12.8 22:06 | OnlyOne | 김영천 | 75.4308
장려 | 5 | 2018.12.14 17:43 | nlp_pln | 이신의, 박장원, 박종성<br>(연세대학교 데이터공학연구실) | 74.8749
장려 | 6 | 2018.12.14 9:17 | kozistr_team | 김형찬 (한국기술교육대학교) | 74.7695

## 참가신청
한국어 자연어 처리에 관심있는 분이라면 누구나 참가 신청할 수 있습니다.
**개인 또는 팀(최대 3명)** 으로 참가 가능합니다. 네이버 폼으로 참가 신청하세요.
**참가신청은 선착순 200팀으로 한정합니다.**
- 신청기간: 2018-11-14에 시작, 200번째 팀 신청 완료 또는 challenge 마감 날짜 1주일 전 종료
- 참가 신청 폼: http://naver.me/IIF5AjtC
  - 참가 동의서: https://github.com/naver/nlp-challenge/blob/master/AGREEMENT.md
  - (중요) **신청 이후 팀명 및 팀원의 추가/삭제/변경은 불가하오니 신청 정보를 잘 확인해 주세요.**
- NSML에 접근하기 위해서는 github 계정이 필요합니다.
- 시상식을 포함한 NLP Workshop은 별도의 [참가신청](https://github.com/naver/nlp-challenge/blob/master/nlp-workshop.md)이 필요합니다.

## (:star: 필독) 참가신청 주의사항
다음 사항을 지켜주시지 않으면 참가 등록에 **불이익**이 발생할 수 있습니다. **반드시** 확인해주세요.
- github username은 **5자리 이상으로 지정**하셔야 합니다.
- Team 이름은 **대/소문자 영어 5~20자리, 특수문자는 “_” 만 사용**하여 지정하셔야 합니다.
- 신청하는 Team 이름과 동일한 github username이 없어야 합니다. 
  Team 이름과 팀원들의 github username은 **모두 다르게 지정**하셔야 합니다.

## NSML 접근 유의사항
- 참가 신청 후 다음 날 오후 1시부터 NSML에 접근 가능합니다. 하루의 참가신청 내역을 취합하여 다음날 오후 1시 이전까지 NSML에 접속할 수 있도록 등록해 드립니다.
  - 단, 금요일 참가 신청 내역은 차주 월요일에 등록됩니다.
- **github 계정의 Two-factor authentication을 해제**해 두어야 로그인이 가능합니다.
  - 확인 방법 : Settings > Security > Two-factor authentication 체크
- github에서 email 주소를 **public**으로 세팅하셔서 NSML 관련 메시지를 수신할 수 있게 해주세요. private로 설정되지 않게 유의해주세요. ( [이메일 설정 도움말](https://help.github.com/articles/setting-your-commit-email-address-on-github/) )

## 일정
- 2018-11-16 12:00:00 시작: 훈련데이터 공개
- 2018-12-15 00:00:00 종료
- 2018-12-28 11:00 우수 참가자 시상식을 포함한 워크샵 (네이버 그린팩토리 2층) [참가신청](https://github.com/naver/nlp-challenge/blob/master/nlp-workshop.md)

## 미션
- [개체명 인식(NER)](http://air.changwon.ac.kr/?page_id=10)
- [의미역 결정(SRL)](http://air.changwon.ac.kr/?page_id=14)

## 진행 방식 및 심사 기준
- 한 개인이 둘 이상의 팀에 포함될 수 없습니다.
  - 만일 피치 못할 사정이 있다면 운영진에 메일을 주세요.
- 참가 가능한 팀은 총 200팀입니다.
  - 참가하는 팀은 두 개의 task 중 어느 것이든, 그리고 양쪽 모두 참가 가능합니다.
- 전체 팀들의 성능은 하루에 1회 업데이트 됩니다.
  - 미션 설명 페이지에서 참가자들의 전체 성능을 확인할 수 있습니다.
- 평가 방식
  - 상세 미션 페이지에서 확인 가능합니다.

## 시상 및 혜택
- 각 Task에 대해 아래와 같이 시상합니다.
  - 대상(1팀): 상금 300만원
  - 우수(2팀): 각 팀 상금 100만원
  - 장려(3팀): 각 팀 상금 50만원
- 우수 참가자는 네이버 채용 지원시에 혜택이 주어집니다.

## 주관/주최
창원대학교 적응지능연구실, 네이버

## FAQ
- [FAQ 페이지](https://github.com/naver/nlp-challenge/blob/master/FAQ.md)

## 문의사항
[Issue 페이지](https://github.com/naver/nlp-challenge/issues)에 Tag를 추가하여 남겨주세요. 담당자가 답변드리겠습니다.

Issue 페이지 문의사항에 대한 답변 운영시간은 월-금 10:00-19:00 입니다.
