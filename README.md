# Gokart Mlflow Sample

これは、GokartとMlflowを組み合わせたサンプルプロジェクトです。

NTCIR-18のSushiデータセットを使った検索タスクをGokart、Mlflowを使ってデータセットの読み込みからモデルの学習、評価までを行います。

基本的に処理の内容はSushiタスクで公開されているサンプルコードを参考に構築されています。

```bash
uv sync # パッケージ管理はuvを使っています

python src/pipeline.py
```
