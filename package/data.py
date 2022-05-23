from locale import normalize
import pandas as pd
import numpy as np
from sklearn import preprocessing

import requests
from bs4 import BeautifulSoup as bs
import bs4
import re
import datetime as dt

from package.race_util import racetimestr2timedelta

import abc

def process_race_id(id, n=1):
    data=dict()
    key_index_dict = {'year':(0,4),'course':(4,6), 'round':(6,8), 'day_of_round':(8,10), 'race_of_day':(10,12)}
    for k,v in key_index_dict.items():
        data[k]= n*[id[v[0]:v[1]]]
    return pd.DataFrame.from_dict(data)

def process_returns(url, n=1):
    df=pd.read_html(url)[1]
    df.columns=['type','result_num','return','popularity']
    df = df.groupby(['type']).agg(tuple).applymap(list).reset_index()
    df['result']= df[[x for x in df.columns if x!='type']].to_dict(orient='records')

    df=df.drop([x for x in df.columns if not x in ['type','result']], axis=1)
    df.index = pd.Index([0]*len(df))
    df = df.pivot(columns=['type'])
    # # res = pd.DataFrame(np.repeat(df.iloc[0,:],n))
    df.columns = df.columns.droplevel()
    res = df.loc[df.index.repeat(n)].reset_index(drop=True)

    return res

def preprocessing(results):
    df = results.copy()

    # 着順に数字以外の文字列が含まれているものを取り除く
    df = df[~(df["着順"].astype(str).str.contains("\\D"))]
    df["着順"] = df["着順"].astype(int)

    # 性齢を性と年齢に分ける
    df["sex"] = df["性齢"].map(lambda x: str(x)[0])
    df["age"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

    # 馬体重を体重と体重変化に分ける
    df["体重"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
    df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

    # データをint, floatに変換
    df["単勝_odd"] = df["単勝_odd"].astype(float)

    # 不要な列を削除
    df.drop(["タイム", "着差", "調教師", "性齢", "馬体重"], axis=1, inplace=True)

    return df

def fetch_race_meta(url, n):
    html = requests.get(url)
    html.encoding = 'EUC-JP'
    soup = bs4.BeautifulSoup(html.text, 'html.parser')
    tag = soup.find('div', attrs={'class': 'data_intro'})
    res = dict()
    if isinstance(tag, bs4.element.Tag):
        infos=re.findall(r'\w+',
                        tag.find_all('p')[0].text + tag.find_all('p')[1].text)
        for info in infos:
            if info in ['芝','ダート','障害']:
                res['field_type'] = info
            if info in ['良','稍重', '重', '不良']:
                res['ground_condition'] = info
            if info in ['曇', '晴', '雨', '小雪', '雪','小雨']:
                res['weather']= info
            if '年' in info and '月' in info:
                res['date']= dt.datetime.strptime(info, '%Y年%m月%d日').date()
        df = pd.DataFrame(res,index=[0])
        df = df.loc[df.index.repeat(n)].reset_index(drop=True)
        return  df
    else:
        raise RuntimeError("can not find the right tag")

def fetch_horse_id(url, n):
    html = requests.get(url)
    html.encoding = 'EUC-JP'
    soup = bs4.BeautifulSoup(html.text, 'html.parser')
    tag = soup.find('div', attrs={'class': 'data_intro'})
    res = dict()
    if isinstance(tag, bs4.element.Tag):
        infos=re.findall(r'\w+',
                        tag.find_all('p')[0].text + tag.find_all('p')[1].text)
        for info in infos:
            if info in ['芝','ダート','障害']:
                res['field_type'] = info
            if info in ['良','稍重', '重', '不良']:
                res['ground_condition'] = info
            if info in ['曇', '晴', '雨', '小雪', '雪','小雨']:
                res['weather']= info
            if '年' in info and '月' in info:
                res['date']= dt.datetime.strptime(info, '%Y年%m月%d日').date()
        df = pd.DataFrame(res,index=[0])
        df = df.loc[df.index.repeat(n)].reset_index(drop=True)
        return  df
    else:
        raise RuntimeError("can not find the right tag")

class Data:

    def __init__(self, id, base_url='https://db.netkeiba.com/') -> None:
        self.id = id
        self.url = f'{base_url}{self.DATA_TYPE()}/{id}'
        html = requests.get(self.url)
        html.encoding = 'EUC-JP'
        self.soup = bs4.BeautifulSoup(html.text, 'html.parser')
        self.tables=pd.read_html(self.url)
        self.data = self.fetch_data()

    @classmethod
    @abc.abstractmethod
    def DATA_TYPE(cls) -> str:
        raise NotImplementedError("Please Implement Data Type")

    @abc.abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        raise NotImplementedError("Please Implement fetch_data")

class RaceData(Data):

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'race'
    
    def fetch_data(self) -> pd.DataFrame:
        df = self.fetch_result()
        df = pd.merge(df, self.fetch_race_meta(), on='race_id')
        return df 

    def append_race_id(self, df:pd.DataFrame) -> pd.DataFrame:
        df['race_id'] = self.id
        return df

    def fetch_result(self):
        df = self.tables[0]
        df = df[~(df["着順"].astype(str).str.contains('\\D'))]
        df["着順"] = df["着順"].astype(int)
        
        df["sex"] = df["性齢"].map(lambda x: str(x)[0])
        df["age"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        df["horse_weight"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
        df["horse_weight_diff"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

        df["単勝"] = df["単勝"].astype(float)

        df.drop(["着差", "調教師", "性齢", "馬体重"], axis=1, inplace=True)
        
        df["タイム"] = df["タイム"].apply(lambda x: racetimestr2timedelta(x))
        #着順,枠番,馬番,馬名,斤量,騎手,単勝,人気	
        col_names=['rank','lane_num','horse_num', 'horse_name']
        col_names+=['load_weight', 'jockey_name', 'time', 'rank_first_odd', 'popularity']
        col_names+=['sex','age','horse_weight', 'horse_weight_diff']
        df.columns=col_names
        normalize_cols = ['horse_weight', 'load_weight']
        df[[f'norm_{x}' for x in normalize_cols]] = df[normalize_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return self.append_race_id(df)

    def fetch_returns(self) -> pd.DataFrame:
        df=self.tables[1]
        df.columns=['type','result_num','return','popularity']
        df = df.groupby(['type']).agg(tuple).applymap(list).reset_index()
        df['result']= df[[x for x in df.columns if x!='type']].to_dict(orient='records')

        df=df.drop([x for x in df.columns if not x in ['type','result']], axis=1)
        df.index = pd.Index([0]*len(df))
        df = df.pivot(columns=['type'])
        df.columns = df.columns.droplevel()        
        return self.append_race_id(df)

    def fetch_race_meta(self) -> pd.DataFrame:
        tag = self.soup.find('div', attrs={'class': 'data_intro'})
        res = dict()
        if isinstance(tag, bs4.element.Tag):
            infos=re.findall(r'\w+',
                            tag.find_all('p')[0].text + tag.find_all('p')[1].text)
            for info in infos:
                if info in ['芝','ダート','障害']:
                    res['field_type'] = info
                if info in ['良','稍重', '重', '不良']:
                    res['ground_condition'] = info
                if info in ['曇', '晴', '雨', '小雪', '雪','小雨']:
                    res['weather']= info
                if '年' in info and '月' in info:
                    res['date']= dt.datetime.strptime(info, '%Y年%m月%d日').date()
            df = pd.DataFrame(res,index=[0])
            return  self.append_race_id(df)
        else:
            raise RuntimeError("can not find the right tag")

    def _fetch_id_from_summary(self, field='horse') -> pd.DataFrame:
        tag = self.soup.find('table', attrs={'summary': 'レース結果'})
        infos=None
        data= dict()
        df = None
        if isinstance(tag, bs4.element.Tag):
            infos=tag.find_all('a', attrs={'href':re.compile(f'^/{field}')})
            for info in infos:
                name = info['title']
                data[name] = re.findall(r'\d+', info['href'])[0]
            df=pd.DataFrame.from_dict(data, orient='index')
            df = df.reset_index(level=0)
        else:
            raise RuntimeError("Tag not fund")
        if isinstance(df, pd.DataFrame):
            raise RuntimeError("Data Frame is None")
        return df.rename(columns={'index': f'{field}_name', 0: f'{field}_id'})

    def fetch_jockey_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='jockey')
    
    def fetch_horse_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='horse')
    
    def fetch_owner_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='owner')
    
    def fetch_trainer_id(self)->pd.DataFrame:
        return self._fetch_id_from_summary(field='trainer')
    
    def get_all_horse_id(self)->list[str]:
        return []


class HorseData(Data):

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'horse'

    def fetch_data(self) -> pd.DataFrame:
        return self.fetch_result()

    def fetch_result(self):
        return self.tables[3]
