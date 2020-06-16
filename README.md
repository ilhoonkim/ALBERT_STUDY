# ALBERT_STUDY
BERT에서 개선된 모델 ALBERT가 나왔습니다.   
BERT에 비해서 ALBERT가 달라진 부분이나 학습 과정에서 여태껏 해왔던 한국어 BERT 학습과 같이 이슈를 수정해보며 한 사이클 돌려보는 것을 목표로 합니다.
## ALBERT 는 무엇이 달라졌는가?
![albert config](https://user-images.githubusercontent.com/45644085/84722469-4a18ae80-afbe-11ea-879f-c030aa6e396b.JPG)   

ALBERT config 파일을 보니 기존 BERT와의 차이를 한눈에 알 수 있었습니다.   
바로 embedding size가 추가된 것인데요. ALBERT의 핵심 중 하나는 바로 이것입니다.   
기존 BERT의 경우는 hidden_size와 embedding_size를 동일하게 가져갔던 것과 달리 ALBERT는 embedding_size를 중간에 개입시켜 파라미터 수를 축소시켰습니다.   

