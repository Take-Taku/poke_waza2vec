# poke_waza2vec
ポケモン対戦において、技構成はポケモンの型を決める上で重要な要素である。
ポケモンの技構成を見ただけでそのポケモンの努力値配分を予測できる場合も少なくない。
また場合によっては、ポケモンの3つの技を見て残り1つの技を推測することも可能である。
このことから、技の役割は周辺の技によって表現されると考えることができる。

今回の手法ではword2vecを用いて
育成されたポケモンのデータからポケモンの技を分散表現で表す。
結果として分散表現で表された技をクラスタリングすることで、
同じ型のポケモンが覚える技同士をある程度分類することができた。
また、技同士のコサイン類似度をとると、
メインウェポンとなるような技は同じタイプの技、
サブウェポンとなるような技は他のサブウェポンとなりうる技が検索結果として現れる傾向が見れた。

![](https://github.com/Take-Taku/poke_waza2vec/issues/1#issue-1315586654)

# DEMO
 
"hoge"の魅力が直感的に伝えわるデモ動画や図解を載せる
 
# Features
 
"hoge"のセールスポイントや差別化などを説明する
 
# Requirement
 
"hoge"を動かすのに必要なライブラリなどを列挙する
 
* huga 3.5.2
* hogehuga 1.0.2
 
# Installation
 
Requirementで列挙したライブラリなどのインストール方法を説明する
 
```bash
pip install huga_package
```
 
# Usage
 
DEMOの実行方法など、"hoge"の基本的な使い方を説明する
 
```bash
git clone https://github.com/hoge/~
cd examples
python demo.py
```
 
# Note
 
注意点などがあれば書く
 
# Author
 
作成情報を列挙する
 
* 作成者
* 所属
* E-mail
 
# License
ライセンスを明示する
 
"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
社内向けなら社外秘であることを明示してる
 
"hoge" is Confidential.
