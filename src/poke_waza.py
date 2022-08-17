from operator import index
import pickle
from pyexpat import model
import re
from unittest import result
from gensim.models.word2vec import Word2Vec

#ラベルエンコーディングするクラス
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import umap

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import cluster
import pprint
import numpy as np
import pickle
import itertools
import os
import json
import glob 



# テキストに出力
def write2txt(dir, results):
    with open(dir, mode='w') as f:
        f.write('\n'.join(results))

def do_PCA(
        vectors
        , dim=-1
        , is_auto=True):      
    pca = PCA()
    feature = pca.fit_transform(vectors)
    if is_auto:
        pca_col = ["PC{}".format(x + 1) for x in range(32)]
        #df_con_ratio = pd.DataFrame([pca.explained_variance_ratio_], columns = pca_col)
        #print('pca raito')
        #print(pca.explained_variance_ratio_)
        raito_sum = 0
        dim = -1
        for i, raito in enumerate(pca.explained_variance_ratio_):
            raito_sum += raito
            if raito_sum > 0.8:
                dim = i+1
                break
        dim = max(2, dim)

    return feature[:, 0:dim]


def do_Umap(
        vectors
        ,dim=2):      
    feature = umap.UMAP(n_components=dim
                        , random_state=42
                        , n_neighbors=15).fit_transform(vectors)

    #pca_col = ["PC{}".format(x + 1) for x in range(self.vector_size)]
    #df_con_ratio = pd.DataFrame([pca.explained_variance_ratio_], columns = pca_col)
    #print(df_con_ratio.head())

    # 指定した次元で返す
    return feature


class Poke_waza2vec():
    NAME = 'waza2vec'
    def __init__(self
                , train_data:str = '../data/RentalPartyData.csv'
                , vector_size:int = 16
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
            self.model_dir = model_dir


    # pickleとしてclassごと保存
    def save(self
            , dir:str ='../model/word2vec/'
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
            , dir:str ='../model/word2vec/'
            , name:str ='model1'):
        waza = cls(model_dir=dir+name)
        return waza


    # word2vecを行う
    def word2vec(self):
        # 技のみをよみこみ
        waza_df = pd.read_csv(self.train_data, index_col=0, na_filter=False)
        waza_df = waza_df.loc[:, ['weapon1','weapon2','weapon3','weapon4']]
        waza_list = waza_df.values.tolist()
        waza_all_list = []
        for waza in waza_list:
            waza_all_list += list(set(itertools.permutations(waza)))
        pprint.pprint('学習データ数: '+str(len(waza_all_list)))

        # word2vecを用いて学習
        model = Word2Vec(
                        waza_all_list,
                        vector_size=self.vector_size,
                        min_count=2,
                        window=3,
                        epochs=100,
                        sg=self.sg,
                        seed=42,
                        workers=1)
        print('学習が終わりました')
        return model


    def create_vector(self, wazas, dim=2, is_auto=True):
        #wazas = self.word2vec.wv.index_to_key   
        vectors = []
        for i, waza in enumerate(wazas):
            waza_vector = self.model.wv.get_vector(waza).tolist()
            vectors.append(waza_vector)

        vectors = do_PCA(vectors, dim, is_auto=is_auto)
        vectors = np.array(vectors)

        return vectors


class Rustic_vec():
    NAME = 'rustic_vec'
    def __init__(self
                , train_data:str = '../data/wazaList.csv'
                , model_dir=None
                ):
        if not model_dir:
            self.train_data = train_data # 学習データのディレクトリ
            self.waza_all_df = self.refomat_df(train_data)

        else:
            with open(model_dir+'/config.json', 'r') as f:
                json_load = json.load(f)
                print(json_load)
            self.train_data = json_load['train_data']
            self.waza_all_df = pd.read_csv(model_dir+'/reshaped.csv', index_col=0)
            self.model_dir = model_dir


    # pickleとしてclassごと保存
    def save(self
            , dir:str ='../model/rustic/'
            , name:str ='model1'):
        if not os.path.exists(dir+name):
            os.makedirs(dir+name)
        
        self.waza_all_df.to_csv(dir+name+'/reshaped.csv', index=False)
        with open(dir+name+'/config.json', 'w') as f:
            config = vars(self)
            del config['waza_all_df']
            json.dump(config, f, indent=2, ensure_ascii=False)


    # 保存したpickleの読み込み
    @classmethod
    def load(cls
            , dir:str ='../model/rustic/'
            , name:str ='model1'):
        waza = cls(model_dir=dir+name)
        return waza


    def refomat_df(self, waza_all_dir):
        def std(df, name):
            value = df[name].values
            value = value.astype('int64')
            return value/value.max()

        df = pd.read_csv(waza_all_dir)
        le = LabelEncoder() 
        df["守る"] = le.fit_transform(df["守る"].values)
        df["直接"] = le.fit_transform(df["直接"].values)
        #df["タイプ"] = le.fit_transform(df["タイプ"].values)
        df["分類"] = le.fit_transform(df["分類"].values)
        df["対象"] = le.fit_transform(df["対象"].values)
        df = df.drop('効果', axis=1)
        df = df.where(df!='-', '-1')


        df['威力']  = std(df, '威力')
        df['ダイ']  = std(df, 'ダイ')
        df['命中']  = std(df, '命中')
        df['PP']  = std(df, 'PP')
        #df['タイプ']  = std(df, 'タイプ')
        df['分類']  = std(df, '分類')
        

        df['威力'] = df['威力'].where(df['威力'] >= 0, -1) 
        df['ダイ'] = df['ダイ'].where(df['ダイ'] >= 0, -1) 
        df['命中'] = df['命中'].where(df['命中'] >= 0, -1)
        df['PP'] = df['PP'].where(df['PP'] >= 0, -1)

        df = pd.get_dummies(df, columns=['タイプ'])
        print(df)
        return df


    def create_vector(self, wazas, dim=2, is_auto=True):
        vectors = []
        waza_size = self.waza_all_df.shape[1]

        for i, waza in enumerate(wazas):
            if waza in ['']:
                waza_vector = np.zeros(waza_size).tolist()
                waza_vector = list(map(float, waza_vector))
                vectors.append(waza_vector)
            else:
                waza_vector = self.waza_all_df.loc[waza, :].values.tolist()
                vectors.append(waza_vector)

        vectors = np.array(vectors)
        vectors = do_PCA(vectors, dim, is_auto=is_auto)
        
        return vectors



class Analyzer:
    def __init__(self
                , models:list = []
                , wazas:list = []
                ):
        self.models = models
        self.wazas = wazas


    def compare_kinds(self):
        df = pd.read_csv('../data/wazaList.csv', index_col=0)
        le = LabelEncoder() 
        df['分類'] = le.fit_transform(df['分類'].values)
        labels = []
        for waza in self.wazas:
            try:
                labels.append(df.at[waza, '分類'])
            except:
                labels.append(-1)

        for model in self.models:
            vectors = model.create_vector(self.wazas)
            output_dir = model.model_dir.replace('model', 'result')
            print(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 2次元で図示
            waza_kinds = ["change", "attack", "spt-at"]
            waza_kinds_id = le.transform(["変化","物理", "特殊"]) 
            #colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
            colorlist = ['r', 'g', 'b']
            markers = [",", "o", "^"]
            for i, id in enumerate(waza_kinds_id):
                index_num = [n for n, v in enumerate(labels) if v == id]
                plt.scatter(
                    vectors[index_num, 0]
                    , vectors[index_num, 1]
                    , c=colorlist[id-1]
                    , label=waza_kinds[i]
                    , s=20
                    , alpha=0.3
                    , marker=markers[i]
                    )

            plt.xlabel("PC1") 
            plt.ylabel("PC2")
            plt.legend()
            plt.savefig(output_dir+'/poke_kinds.png')
            plt.clf()


    def compare_types(self):
        df = pd.read_csv('../data/wazaList.csv', index_col=0)
        le = LabelEncoder() 

        waza_types = list(set(df['タイプ'].values))
        waza_trans = {
            'ノーマル': 'normal'
            ,'ほのお': 'fire'
            ,'みず': 'water'
            ,'でんき': 'electric'
            ,'くさ': 'grass'
            ,'こおり': 'ice'
            ,'かくとう': 'fighting'
            ,'どく': 'poison'
            ,'じめん': 'ground'
            ,'ひこう': 'flying'
            ,'エスパー': 'psychic'
            ,'むし': 'bug'
            ,'いわ': 'rock'
            ,'ゴースト': 'ghost'
            ,'ドラゴン': 'dragon'
            ,'あく': 'dark'
            ,'はがね': 'steel'
            ,'フェアリー': 'fairy'
        }

        df['タイプ'] = le.fit_transform(df['タイプ'].values)
        labels = []
        for waza in self.wazas:
            try:
                labels.append(df.at[waza, 'タイプ'])
            except:
                labels.append(-1)

        for model in self.models:
            vectors = model.create_vector(self.wazas)
            output_dir = model.model_dir.replace('model', 'result')
            print(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 2次元で図示
            waza_types_id = le.transform(waza_types) 
            #cm = plt.get_cmap("Spectral")
            #colorlist = ['r', 'g', 'b']

            print(waza_types_id)
            for i, id in enumerate(waza_types_id):
                index_num = [n for n, v in enumerate(labels) if v == id]

                plt.scatter(
                    vectors[index_num, 0]
                    , vectors[index_num, 1]
                    , color = cm.hsv(i/20.0)
                    , label=waza_trans[waza_types[i]]
                    , s=20
                    , alpha=0.9
                    )

            plt.xlabel("PC1") 
            plt.ylabel("PC2")
            lg = plt.legend(
                    bbox_to_anchor=(1.05, 1)
                    , loc='upper left'
                    , borderaxespad=0
                    , fontsize=10)
            plt.savefig(output_dir+'/poke_types.png'
                    , bbox_extra_artists=(lg,)
                    , bbox_inches='tight'
                    )
            plt.clf()


    def compare_search(self, waza='かえんほうしゃ'):
        for model in self.models:
            vectors = model.create_vector(self.wazas, dim=-1, is_auto=False)
            target_vec =  vectors[self.wazas.index(waza)]
            vec_sim = []
            for i, vector in enumerate(vectors):
                cos_sim = np.dot(vector, target_vec) / \
                    (np.linalg.norm(vector)*np.linalg.norm(target_vec))
                #cos_sim = round(cos_sim, 5)
                vec_sim.append([cos_sim, self.wazas[i]])

            vec_sim = sorted(vec_sim, key=lambda x: x[0])[::-1]
        
            print(model.NAME+'  検索した技: '+waza)
            for sim, w in vec_sim[:6]:
                print(w+'\t'+str(sim))         
            print('-'*20)


if __name__ == '__main__':
    '''
    poke_waza2vec = Poke_waza2vec(
                    train_data='../data/RentalPartyData.csv'
                    , vector_size=32
                    , sg=0
                    )
    rustic_vec = Rustic_vec(
                    train_data='../data/wazaList.csv'
                    )

    poke_waza2vec.save(name='vec1')
    rustic_vec.save(name='vec1')
    '''

    poke_waza2vec = Poke_waza2vec.load(name='vec1')
    rustic_vec = Rustic_vec.load(name='vec1')
    analyzer = Analyzer(
        models=[poke_waza2vec, rustic_vec]
        , wazas=poke_waza2vec.model.wv.index_to_key
        )
    #ana.compare_kinds()
    #ana.compare_types()
    analyzer.compare_search('かえんほうしゃ')
    analyzer.compare_search('たきのぼり')
    analyzer.compare_search('バークアウト')
    analyzer.compare_search('じこさいせい')
    analyzer.compare_search('りゅうのまい')

 