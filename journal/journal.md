# 2024-11-03

No1
モンテカルロ木探索をGPUを用いて実行するようにした。
性能向上や高速化によりなんとか良くなってきたがまだ実行には時間がかかる。
次回はディープラーニングを作成する。

## 結果(raw)

> MonteCarloAI(2) by MonteCarloAI(2) B:0, W:10, Draw:0
> MonteCarloAI(2) by MonteCarloAI(3) B:2, W:8, Draw:0
> MonteCarloAI(2) by MonteCarloAI(4) B:0, W:9, Draw:1
>
> MonteCarloAI(3) by MonteCarloAI(2) B:7, W:3, Draw:0
> MonteCarloAI(3) by MonteCarloAI(3) B:4, W:6, Draw:0
> MonteCarloAI(3) by MonteCarloAI(4) B:8, W:2, Draw:0
>
> MonteCarloAI(4) by MonteCarloAI(2) B:5, W:5, Draw:0
> MonteCarloAI(4) by MonteCarloAI(3) B:3, W:7, Draw:0
> MonteCarloAI(4) by MonteCarloAI(4) B:5, W:4, Draw:1

# 2024-11-04

No2
ディープラーニングを用いたAIを開発した。
initial trainingとself-play trainingの二段階に分けてトレーニングし、LOSSが目標値を下回ったら学習終了するようにした。評価関数はevaluator4を使用した。
結果を見るとわかるように、MinimaxAI(4)に勝利しているのにもかかわらずMinimaxAI(2)やMinimaxAI(3)には完敗しており、
評価関数に応じて先手必勝・後手必勝があるのかもしれない。少なくとも2、3、4の評価関数では正確に計測できていないと考える。
そしてそこに法則があるのでそれを検証する必要がある。

## 結果(raw)

> MaxAI by DeepLearningAI B:0, W:1000, Draw:0
> CornerAI by DeepLearningAI B:244, W:727, Draw:29
> MinimaxAI(2) by DeepLearningAI B:1000, W:0, Draw:0
> MinimaxAI(3) by DeepLearningAI B:1000, W:0, Draw:0
> MinimaxAI(4) by DeepLearningAI B:0, W:1000, Draw:0

# 2024-11-05

No3
評価関数をディープラーニングで実装しようとした。
実装にミスが多く完成はしなかった。

# 2024-11-12

No4
新しい仮説の追加：
ディープラーニングはデータの質が良ければいいほど強くなる

# 2025-01-24

No5
AIを作った。とりあえず5000エピソードで学習。昨日の夜学習始めて15時間かかった。
今日が午前授業だったのでちょうどよかったが、時間がかかりすぎる。しかも、対戦させてみたら弱い。Minimax法にボコボコにされていた。100回戦って100回負けていた。
自分とも対戦してみたが弱かった。こいつには負ける気がしない。Minimax法を対戦相手として学習させたら良さそうなのと、たぶん学習回数が足りないのでもっと学習回数を増やす。GPUを使って高速化する。
