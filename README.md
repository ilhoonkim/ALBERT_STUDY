# ALBERT_STUDY
BERT에서 개선된 모델 ALBERT가 나왔습니다.   
BERT에 비해서 ALBERT가 달라진 부분이나 학습 과정에서 여태껏 해왔던 한국어 BERT 학습과 같이 이슈를 수정해보며 한 사이클 돌려보는 것을 목표로 합니다.
## ALBERT 는 무엇이 달라졌는가?
![albert config](https://user-images.githubusercontent.com/45644085/84722469-4a18ae80-afbe-11ea-879f-c030aa6e396b.JPG)   

### 1) Factorization of embedding parameters
ALBERT config 파일을 보니 기존 BERT와의 차이를 한눈에 알 수 있었습니다.   
바로 embedding size가 추가된 것인데요. ALBERT의 핵심 중 하나는 바로 이것입니다.   
기존 BERT의 경우는 hidden_size와 embedding_size를 동일하게 가져갔던 것과 달리 ALBERT는 embedding_size를 중간에 개입시켜 파라미터 수를 축소시켰습니다.

![그림1](https://user-images.githubusercontent.com/45644085/84724456-391e6c00-afc3-11ea-8107-9c4eb8be27c7.png)   

BERT의 경우 사전의 크기(사전 단어의 갯수)가 꽤 많은 편인데요. 이런 사전 단어들을 임베딩하는데 꽤나 큰 파라미터 사이즈가 발생하게 됩니다.   
ALBERT에서는 V(사전 크기)를 바로 H(hidden layer size)로 임베딩 시키는 것이 아니라 보다 작은 E(embedding size)로 매핑을 한 후에 E 벡터를 다시 H로 보내게 됩니다.    
이렇게 되면 기존 V X H 였던 파라미터 사이즈가 VXE + EXH 로 줄어들게 됩니다.   
BERT는 사전 단어의 갯수가 30000개라고 치면 히든 레이어가 768개일 때 파라미터 사이즈는 23,040,000개나 되는데 반면 ‬ALBERT는 30000 X 128 + 128 X 768 = 3,938,304‬ 개의 파라미터 사이즈를 가지게 됩니다.
