# ALBERT_STUDY
BERT에서 개선된 모델 ALBERT가 나왔습니다.   
BERT에 비해서 ALBERT가 달라진 부분이나 학습 과정에서 여태껏 해왔던 한국어 BERT 학습과 같이 이슈를 수정해보며 한 사이클 돌려보는 것을 목표로 합니다.   
[구글 ALBERT(https://github.com/google-research/ALBERT) 참고](https://github.com/google-research/ALBERT)   

## ALBERT 는 무엇이 달라졌는가?
![albert config](https://user-images.githubusercontent.com/45644085/84722469-4a18ae80-afbe-11ea-879f-c030aa6e396b.JPG)   

### 1) Factorization of embedding parameters
ALBERT config 파일을 보니 기존 BERT와의 차이를 한눈에 알 수 있었습니다.   
바로 embedding size가 추가된 것인데요. ALBERT의 핵심 중 하나는 바로 이것입니다.   
기존 BERT의 경우는 hidden_size와 embedding_size를 동일하게 가져갔던 것과 달리 ALBERT는 embedding_size를 중간에 개입시켜 파라미터 수를 축소시켰습니다.

![84724456-391e6c00-afc3-11ea-8107-9c4eb8be27c7](https://user-images.githubusercontent.com/45644085/85256628-8442ee00-b49f-11ea-8052-4c1e50ed4b99.png)


BERT의 경우 사전의 크기(사전 단어의 갯수)가 꽤 많은 편인데요. 이런 사전 단어들을 임베딩하는데 꽤나 큰 파라미터 사이즈가 발생하게 됩니다.   
ALBERT에서는 V(사전 크기)를 바로 H(hidden layer size)로 임베딩 시키는 것이 아니라 보다 작은 E(embedding size)로 매핑을 한 후에 E 벡터를 다시 H로 보내게 됩니다.    
이렇게 되면 기존 VⅩH 였던 파라미터 사이즈가 VⅩE + EⅩH 로 줄어들게 됩니다.   
BERT는 사전 단어의 갯수가 30000개라고 치면 히든 레이어가 768개일 때 파라미터 사이즈는 23,040,000개나 되는데 반면 ‬ALBERT는 30000 X 128 + 128 X 768 = 3,938,304‬ 개의 파라미터 사이즈를 가지게 됩니다.


### 2) Cross-layer parameter sharing

ALBERT는 Layer 간 파라미터를 공유합니다.   

<img src = "https://user-images.githubusercontent.com/45644085/84725316-4d636880-afc5-11ea-80af-98db8e57da5f.png" align = "center">   

Layer 간에 파라미터를 공유한다고 하더라도 모델 성능이 거의 떨어지지 않는다는 것은 논문 실험결과를 통해 검증이 되어 있습니다. 그 이유에 대해서는 자세히 설명되어 있지는 않아 조금 더 공부를 할 필요가 있습니다.   
이러한 방식때문에 ALBERT는 Layer 갯수만큼 파라미터 사이즈가 늘어나는 일이 없게 되었습니다.   
Factorization of embedding parameters과 Cross-layer parameter sharing의 방식으로 인해 최종적인 파라미터 사이즈는 BASE 모델 기준으로 BERT는 110M , ALBERT는 12M 으로 9배의 차이가 납니다.

### 3) SOP(Sentence Order Prediction) loss
![그림1](https://user-images.githubusercontent.com/45644085/84726625-468a2500-afc8-11ea-9ed8-118f66cc7403.png)   

기존의 BERT는 NSP(Next sentence prediction)을 학습하였는데요. 필자를 포함하여 BERT를 써본 사람들은 보통 이것이 크게 의미있다고 느끼지 못했던 것 같습니다. 또한 다양한 논문에서도 NSP의 효과에 대해 의문을 제기했다고도 하고요. 이에 다음 문장을 예측하는 것보다는 문장의 일관성을 예측하는 문제로 변형하여 학습을 하게 되었습니다.   
기존 BERT에는 프리트레이닝 데이터를 만들 때 연속된 세그먼트와 랜덤한 두 세그먼트를 가져와서 연속된 문장인지 바이너리로 예측하는 형태였다면 ALBERT에서는 연속된 두 세그먼트를 가져와서 50% 확률로 순서를 뒤집어서 모델을 통해 순서가 제대로 된건지 예측하도록 합니다.

<img width="759" alt="fig5" src="https://user-images.githubusercontent.com/45644085/84726735-7fc29500-afc8-11ea-831e-8549579777a0.png">   
실제로 NSP를 목표로 학습한 경우 SOP에서 상당히 성능이 떨어지지만 SOP를 학습한 경우 NSP에서도 좋은 성능을 보인다고 합니다.   

---   

## ALBERT 학습 과정   
간단하게 ALBERT가 기존의 BERT와 달라진 점을 알아보았으니 실제로 BERT와 같이 하나하나 진행하려고 합니다.   
### 1.Tokenization 
ALBERT에서는 기존 tokenization과는 달리 .txt 확장자의 사전 파일이 아니라 spm 모델을 input으로 하여 사전으로 사용할 수 있다는 차이가 생겼습니다. 하지만 spm 모델 자체가 가지는 사전은 제 깃헙  BERT Tokenizer 편에서 언급했듯이 바로 사용하기엔 문제가 있다고 생각하여 저는 spm 모델은 사용하지 않고 기존 BERT 방식대로 이미 만들어진 사전 파일을 사용하기로 했고 필요없는 spm 관련 코드는 정리해버림과 동시에 한글 토크나이징에 최적화 되도록 변경하였습니다. 이에 대한 내용은 [제 BERT Tokenizer 깃헙편](https://github.com/ilhoonkim/BertTokenizer)에서 확인할 수 있습니다.   
**tokenization_ns_kor.py** 참조   

### 2.프리트레이닝 데이터 만들기   
사전학습 파일을 만드는 create_pretraining_data의 경우에도 spm 관련 코드가 존재하였는데 마찬가지로 해당 부분은 사용하지 않기로 하여 주석처리 하였습니다. 이외에 다른 부분은 변경하지 않았습니다.   
**Create_pretraining_data_ns.py** 참조

