# Faster-R-CNN

torchvisionのチュートリアルをベースにしたfaster r-cnn

BDD100KをPascal VOCフォーマットに変換して使用


- dataloader.py データ作成
- train.py      学習
- test.py       画像を入力して結果を表示

# 使い方
train.pyにデータセットのディレクトリ指定して実行
test.pyのモデルのロード部分で書き出したモデルを読み込んで推論
