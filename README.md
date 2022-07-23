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

![参考図](https://user-images.githubusercontent.com/68584494/180594737-85479df4-1a8a-48ab-8e7d-83e39b031d79.png)

# Usage
 
実行方法

実行したい内容によって適宜poke_waza.pyの
if __name__ == '__main__':
以下の部分を変更するよよい。
 
```bash
git clone https://github.com/Take-Taku/poke_waza2vec.git
cd poke_waza2vec
virtualenv -p python3.8.1 venv
source venv/bin/activate
pip install -r requirements.txt
cd src
python poke_waza.py
```
 
# Note
Wazaクラスのメソッドについて簡単に説明
  
モデルの保存
```bass
Waza.save()
```
モデルの読み込み
```bass
Waza.load()
```
word2vecによるモデルの学習
```bass
Waza.word2vec()
```
入力した技に一番近いcos類似度の技を返す
```bass
Waza.similarity('waza')
```
複数の入力した技の計算を行い、一番近いcos類似度の技を返す
word2vecを用いているのでついでに実装したが、解釈が困難
```bass
Waza.similarity(pos=['waza1', 'waza2], neg=['waza3])
```
pcaで次元圧縮した後、kmeansを用いてクラスタリングを行う。
結果を図とグループされた技ごとに出力する

 
# License

