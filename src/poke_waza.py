import pickle
from pyexpat import model
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
import pprint
import numpy as np
import pickle
import itertools
import os
import json


# テキストに出力
def write2txt(dir, results):
    with open(dir, mode='w') as f:
        f.write('\n'.join(results))

class Waza():
    def __init__(self
                , train_data:str = '../data/RentalPartyData.csv'
                , vector_size:int = 32
                , sg:int = 0
                , model_dir=None
                ):
        if not model_dir:
            self.train_data = train_data # 学習データのディレクトリ
            self.vector_size = vector_size # word2vecのベクトルの次元 
            self.sg = sg #cbow 0  skip-gram 1
            self.model = self.word2vec()

        else:
            with open(model_dir+'/config.json', 'r') as f:
                json_load = json.load(f)
                print(json_load)
            self.train_data = json_load['train_data']
            self.vector_size = json_load['vector_size']
            self.sg = json_load['sg']
            self.model = Word2Vec.load(model_dir+'/poke_waza2vec.model')

    # pickleとしてclassごと保存
    def save(self
            , dir:str ='../model/'
            , name:str ='model1'):
        if not os.path.exists(dir+name):
            os.makedirs(dir+name)
        
        self.model.save(dir+name+'/poke_waza2vec.model')
        write2txt(dir+name+'/wazaList.txt', self.model.wv.index_to_key) 
        with open(dir+name+'/config.json', 'w') as f:
            config = vars(self)
            del config['model']
            
            json.dump(config, f, indent=2, ensure_ascii=False)


    # 保存したpickleの読み込み
    @classmethod
    def load(cls
            , dir:str ='../model/'
            , name:str ='model1'):
        waza = cls(model_dir=dir+name)
        return waza
        
    # word2vecを行う
    def word2vec(self):
        # 技のみをよみこみ
        waza_df = pd.read_csv(self.train_data, index_col=0, na_filter=False)
        waza_df = waza_df.loc[:, ['weapon1','weapon2','weapon3','weapon4']]
        waza_list = waza_df.values.tolist()
        
        # word2vecを用いて学習
        model = Word2Vec(
                        waza_list,
                        vector_size=self.vector_size,
                        min_count=1,
                        window=3,
                        epochs=100,
                        sg=self.sg,
                        seed=42,
                        workers=1)
        print('学習が終わりました')
        return model


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
    
    def wazas_similarity(self, wazas):
        sim = self.model.wv.similarity(*wazas)
        return sim
 


if __name__ == '__main__':
    
    waza = Waza(train_data='../data/RentalPartyData.csv'
                , vector_size=24
                , sg=0)
    
    waza.save(name='model1')
    
    waza = Waza.load(name='model1')
  
    waza.similarity(pos=['じしん', 'ねっとう'], neg=['たきのぼり'])    
    waza.similarity('アクアブレイク')    
    
    waza.do_kmeans(n_clusters=3, dim=2)
    #waza.poke_waza_sim()
