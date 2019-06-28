# Motif

여러가지 데이터 셋들로 suco가 얼마나 정확한지 manual하게 보고자 하였다. feature extraction 으로는 librosa 라이브러리에서 제공되는 chromagram_cens를 사용하여 0초~90초의 데이터만을 활용했다.

# with good dataset

![suco](https://user-images.githubusercontent.com/34101059/60353469-0a4f6480-9a05-11e9-9894-a985cac233ad.png)

반주도, 성별도 같은 <구애> (선우정아) 커버곡과 이들과 아예 다른 장르인 힙합, 팝 곡들을 포함하여 suco를 돌렸을 때이다.

- [원곡](https://www.youtube.com/watch?v=4FLGSN1RG5A)
- [커버곡](https://www.youtube.com/watch?v=bxonRaCn4fY)

좋은 결과가 나온다.

# with tricky cover songs

![suco_more](https://user-images.githubusercontent.com/34101059/60353647-70d48280-9a05-11e9-8be9-6d5722b7b72f.png)

이번엔 아까의 케이스에 더해 김장훈의 <나와 같다면> 원곡과 여러 커버곡들을 준비해보았다. 

- [원곡](https://www.youtube.com/watch?v=yKPEb5-RP0M)
- [parody](https://www.youtube.com/watch?v=UeoXODwwT4A)
- [cover1](https://www.youtube.com/watch?v=Lt8QqNVc3Jc)
- [cover2](https://www.youtube.com/watch?v=-aur9I9FJwQ) (사실 이게 원곡)

parody는 구조적 차이도 있고, 노래방이라 잔향도 심하고, 잡음도 많다. 게다가 음악도 과장하게 불렀지만 인간 식역으로는 김장훈의 <나와 같다면>을 부른 것을 충분히 알 수 있어서 혹시 인식할 수 있지 않을까 기대했던 노래이다. cover1과 cover2는 원곡과 비교하여 반주도, 구조적 차이도 조금 있다. cover1 같은 경우는 나가수에서 부른 것이다 보니 중간에 애드리브, 평론가 인터뷰 또한 있다. **이런 케이스에 대해선 결과가 좋지 못한 것을 확인할 수 있다.**

# extensive songs

앞서 케이스에서 너무 미진한 결과를 보여 좀 더 많은 오픈된 [데이터셋](https://sites.google.com/site/ismir2015shapelets/home)을 사용해보았다.

![moreExtensive](https://user-images.githubusercontent.com/34101059/60354088-6d8dc680-9a06-11e9-90a0-5f52ce817ac4.png)

같은 노래 제목을 가진 것들 중 하나는 원곡이고 다른 하나가 커버곡인 것으로 보인다. 그래서 대각선 방향으로 2 X 2 box가 진하면 잘 되는 결과라 해석할 수 있는데 어느정도 그렇다. 대략 거리가 10정도에서 cover 곡 관계가 생기는 것 같다. 