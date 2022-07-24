import pickle
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
import pprint
import numpy as np
import pickle
<<<<<<< HEAD
import itertools

=======
>>>>>>> b7c2157f0d9222f876faa7dae60384d0e087fc2f

# テキストに出力
def write2txt(dir, results):
    with open(dir, mode='w') as f:
        f.write('\n'.join(results))

class Waza():
    def __init__(self
                , train_data:str = '../data/nurturedpoke.csv'
<<<<<<< HEAD
                , vector_size:int = 32
=======
                , vector_size:int = 16
>>>>>>> b7c2157f0d9222f876faa7dae60384d0e087fc2f
                , sg:int = 0
                ):
        self.train_data = train_data # 学習データのディレクトリ
        self.vector_size = vector_size # word2vecのベクトルの次元 
        self.sg = sg #cbow 0  skip-gram 1

    # pickleとしてclassごと保存
    def save(self
            , dir:str ='../data/'
            , name:str ='waza.pickle'):
        with open(dir+name, 'wb') as f:
            pickle.dump(self, f)
        print('保存しました')

    # 保存したpickleの読み込み
    @classmethod
    def load(cls
            , dir:str ='../data/'
            , name:str ='waza.pickle'):
        with open(dir+name, 'rb') as f:
            waza = pickle.load(f)
        return waza
        
    # word2vecを行う
    def word2vec(self):
        # 技のみをよみこみ
        waza_df = pd.read_csv(self.train_data, index_col=0)
        waza_list = waza_df.values.tolist()

        # word2vecを用いて学習
        self.model = Word2Vec(
                        waza_list,
                        vector_size=self.vector_size,
<<<<<<< HEAD
                        min_count=1,
                        window=3,
                        epochs=100,
                        sg=self.sg,
                        seed=42,
                        workers=1)
=======
                        min_count=5,
                        window=3,
                        epochs=100,
                        sg=self.sg)
>>>>>>> b7c2157f0d9222f876faa7dae60384d0e087fc2f
        write2txt('../data/wazaList.txt', self.model.wv.index_to_key) 
        print('学習が終わりました')

    # 入力された技と近い技を出力
    # デフォルトは火炎放射
    def similarity(self
                    , pos=[]
                    , neg=[]
                    ):
        if not pos and not neg:
            pos.append('かえんほうしゃ')
        if type(pos) is str:
            pos = [pos]

        waza_sim = None
        try:
            if not pos:
                waza_sim = self.model.wv.most_similar(negative=neg)
            if not neg:
                waza_sim = self.model.wv.most_similar(positive=pos)
            else:
                waza_sim = self.model.wv.most_similar(positive=pos, negative=neg)

        except:
            print('技を確認してください')

        # 出力
        pprint.pprint(waza_sim)

    # pcaを行う
    def do_PCA(self
                ,dim=2):      
        pca = PCA()
        pca.fit(self.get_waza_vector())
        feature = pca.transform(self.get_waza_vector())

        #pca_col = ["PC{}".format(x + 1) for x in range(self.vector_size)]
        #df_con_ratio = pd.DataFrame([pca.explained_variance_ratio_], columns = pca_col)
        #print(df_con_ratio.head())

        # 指定した次元で返す
        return feature[:, 0:dim]

    #  クラスタリングを行う
    def do_kmeans(self
                , n_clusters=3
                , dim=2):
        # 特徴量を圧縮
        feature = self.do_PCA(dim=dim)
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(feature)        
        labels = kmeans.labels_

        # 2次元で図示
        plt.scatter(feature[:, 0], feature[:, 1], c=labels)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig('../fig/kmeans.png')

        for num in range(n_clusters):
            same_group = []
            for i, r in enumerate(labels):
                if num == r:
                    same_group.append(self.model.wv.index_to_key[i])  
                write2txt('../result/group'+str(num)+'.txt', same_group) 

    # 技をベクトルで表す
    def get_waza_vector(self):
        wazas = self.model.wv.index_to_key   
        vectors = []
        for waza in wazas:
            waza_vector = self.model.wv.get_vector(waza)
            vectors.append(waza_vector)
        return vectors
    
<<<<<<< HEAD
    def wazas_similarity(self, wazas):
        sim = self.model.wv.similarity(*wazas)
        return sim

    def poke_waza_sim(self):
        # 技のみをよみこみ
        waza_df = pd.read_csv('../../../../Desktop/nurturedpoke.csv', index_col=0)
        waza_list = waza_df.values.tolist()
        poke = []
        for waza in waza_list:
            sum = 0
            for pair in itertools.combinations(waza, 2):
                sum += self.wazas_similarity(list(pair))
            poke.append([waza, sum])

        poke = sorted(poke, key=lambda x: x[1])
        poke_v = [p[1] for p in poke]
        pprint.pprint(poke[300:500])
        
        plt.scatter(list(range(len(poke))), poke_v)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        #plt.show()


if __name__ == '__main__':
    waza = Waza(train_data = '../../../../Desktop/nurturedpoke.csv'
                , vector_size = 32
                , sg=0)
    waza.word2vec()
    waza.save(name='waza.pickle')
    waza = Waza.load()
  
    waza.similarity('ねっとう')   
    waza.similarity('ほうでん')    
    waza.similarity('フレアドライブ')    
    #waza.do_kmeans()
    #waza.poke_waza_sim()
=======

if __name__ == '__main__':
    waza = Waza(train_data = '../data/nurturedpoke.csv'
                , vector_size = 32)
    waza.word2vec()
    waza.save(name='waza.pickle')
    waza = Waza.load()
    waza.similarity('おにび')
    waza.do_kmeans()
>>>>>>> b7c2157f0d9222f876faa7dae60384d0e087fc2f
