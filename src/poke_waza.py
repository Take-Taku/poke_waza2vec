import pickle
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
import pprint
import numpy as np
import pickle

def write2txt(dir, results):
    with open(dir, mode='w') as f:
        f.write('\n'.join(results))

class Waza():
    def __init__(self
                , train_data = '../data/nurturedpoke.csv'
                , vector_size = 16
                , sg = 0 #cbow 0  skip-gram 1
                ):
        self.train_data = train_data
        self.vector_size = vector_size
        self.sg = sg

    # 保存
    def save(self
            , dir='../data/'
            , name='waza.pickle'):
        with open(dir+name, 'wb') as f:
            pickle.dump(self, f)
        print('保存しました')

    # 読み込み
    @classmethod
    def load(cls
            , dir='../data/'
            , name='waza.pickle'):
        with open(dir+name, 'rb') as f:
            waza = pickle.load(f)
        return waza
        
    # word2vecを行う
    def word2vec(self):
        poke_df = pd.read_csv(self.train_data, index_col=0)
        waza_df = poke_df['4']
        waza_list = waza_df.tolist()
        waza_list = [w.split('/') for w in waza_list]

        waza_all = []
        for waza in waza_list:
            waza_all += waza
            waza_all = list(set(waza_all))

        self.model = Word2Vec(
                        waza_list,
                        vector_size=self.vector_size,
                        min_count=5,
                        window=3,
                        epochs=100,
                        sg=self.sg)
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

        pprint.pprint(waza_sim)


    def do_PCA(self
                ,dim=2):      
        pca = PCA()
        pca.fit(self.get_waza_vector())
        feature = pca.transform(self.get_waza_vector())

        #pca_col = ["PC{}".format(x + 1) for x in range(self.vector_size)]
        #df_con_ratio = pd.DataFrame([pca.explained_variance_ratio_], columns = pca_col)
        #print(df_con_ratio.head())
        return feature[:, 0:dim]

  
    def do_kmeans(self
                , n_clusters=3
                , dim=2):
        feature = self.do_PCA(dim=dim)
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(feature)        
        labels = kmeans.labels_

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


    def get_waza_vector(self):
        wazas = self.model.wv.index_to_key   
        vectors = []
        for waza in wazas:
            waza_vector = self.model.wv.get_vector(waza)
            vectors.append(waza_vector)
        return vectors
    

if __name__ == '__main__':
    #waza = Waza(train_data = '../data/nurturedpoke.csv'
    #            , vector_size = 32)
    #waza.word2vec()
    #waza.save(name='waza.pickle')
    waza = Waza.load()
    waza.similarity('おにび')
    waza.do_kmeans()