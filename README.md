# poke_waza2vec
ポケモン対戦において、技構成はポケモンの型を決める上で重要な要素である。
ポケモンの技構成を見ただけでそのポケモンの努力値配分を予測できる場合も少なくない。
また場合によっては、ポケモンの3つの技を見て残り1つの技を推測することも可能である。
このことから、技の役割は周辺の技によって表現されると考えることができる。

今回の手法ではword2vecを用いて
育成されたポケモンのデータからポケモンの技を分散表現で表す。
結果として分散表現で表された技をクラスタリングすることで、
同じ型のポケモンが覚える技同士をある程度分類することができた。

一方で、ベクトルの類似度をとると、若干ではあるが、よく一緒に採用される技や
同系統の技が予測される傾向がみれた。

![参考図](https://user-images.githubusercontent.com/68584494/185729022-8b2e6ce8-345e-4e32-b122-076745584b98.png)

# Data
Fusicの塚本という方がデータを置いてくれていましたので、そちらを使用いたしました。

ブログ：https://tech.fusic.co.jp/posts/2020-11-26-pokemon-prog/

データ元：https://gist.github.com/TsuMakoto/c465f82513b3ab3bf98ca0bed7fe1936


# Usage
 
実行方法

実行したい内容によって適宜poke_waza.pyの
 if __name__ == '__main__': 
以降の部分を変更するとよい。
 
```bash
git clone https://github.com/Take-Taku/poke_waza2vec.git
cd poke_waza2vec
virtualenv -p python3.8.1 venv
source venv/bin/activate
pip install -r requirements.txt
cd src
python poke_waza.py
```

