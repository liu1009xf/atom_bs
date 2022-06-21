from typing import Any, Optional
import pandas as pd
import numpy as np
from sklearn import preprocessing
from copy import deepcopy
import numbers
import requests
from bs4 import BeautifulSoup as bs
import bs4
import re
import datetime as dt

from package.race_util import racetimestr2timedelta

import abc

from logging import getLogger
logger = getLogger(__name__)
from tqdm.notebook import tqdm

'''
e.g.:
race_ids = get_race_id_list(dt.datetime(2016,1,1),dt.datetime(2016,12,31))
store= pd.HDFStore('~/research/horse_racing_forecast/data/data.h5')
store.append(key='race_id', value= pd.DataFrame({"race_id":race_ids, 'year':2016}), format='t', data_columns=True)
'''
def get_race_id_list(start, end,):
    res = list()
    for d in tqdm(start + dt.timedelta(n) for n in range((end-start).days)):
        res+=get_race_id_list_from_date(d)
    return res

def get_race_id_list_from_date(today):
    date = f'{today.year:04}{today.month:02}{today.day:02}'
    url = 'https://db.netkeiba.com/race/list/' + date
    html = requests.get(url)
    html.encoding = "EUC-JP"
    soup = bs4.BeautifulSoup(html.text, "html.parser")
    race_list = soup.find('div', attrs={"class": 'race_list fc'})
    if race_list is None:
        return list()
    a_tag_list = race_list.find_all('a')  # type: ignore
    href_list = [a_tag.get('href') for a_tag in a_tag_list]
    race_id_list = list()
    for href in href_list:
        for race_id in re.findall('[0-9]{12}', href):
            race_id_list.append(race_id)
    return list(set(race_id_list))

def race_ids_by_year(year:int):
    race_id_list = []
    for place in range(1, 11, 1):
        for kai in range(1, 6, 1):
            for day in range(1, 13, 1):
                for r in range(1, 13, 1):
                    race_id = str(year) + str(place).zfill(2) + str(kai).zfill(2) +\
                    str(day).zfill(2) + str(r).zfill(2)
                    race_id_list.append(race_id)
    return race_id_list

class Data:
    def __init__(self, 
                id:str, 
                base_url:str='https://db.netkeiba.com/',  
                cache = False,
                store:Optional[pd.HDFStore] = None) -> None:
        self.id = id
        self.cache_loaded  = False
        self.cache_loaded=False
        if cache and store != None and f'/{self.PATH_SUFFIX()}' in store.keys():
            try:
                self.data = pd.read_hdf(store, key=self.DATA_TYPE(),table=True, mode='r', where=[f'{self.ID_PREFIX()}_id=\"{id}\"'])
                self.cache_loaded  = not self.data.empty
            except:
                logger.warning("cannot load data from store!")
                pass

        if not self.cache_loaded :
            self.url = f'{base_url}{self.PATH_SUFFIX()}/{id}'
            html = requests.get(self.url)
            html.encoding = 'EUC-JP'
            self.soup = bs4.BeautifulSoup(html.text, 'html.parser')
            self.tables=pd.read_html(self.url)
            self.data = self.fetch_data()
        
        self.data = self.data.astype(self.COL_TYPES())
        if not self.cache_loaded  and cache and store != None:
            store.append(key=self.DATA_TYPE(), value= self.data, format='t', data_columns=True, min_itemsize=self.COL_SIZE())  # type: ignore
        
        self.process_data()


    @classmethod
    @abc.abstractmethod
    def DATA_TYPE(cls) -> str:
        raise NotImplementedError("Please Implement Data Type")

    @classmethod
    def PATH_SUFFIX(cls)->str:
        return cls.DATA_TYPE()
    
    @classmethod
    def ID_PREFIX(cls)->str:
        return cls.DATA_TYPE()

    @classmethod
    @abc.abstractmethod
    def COL_TYPES(cls) -> str:
        raise NotImplementedError("Please Implement Data Type")

    @classmethod
    @abc.abstractmethod
    def COL_SIZE(cls) -> dict:
        return {}
    
    @abc.abstractmethod
    def process_data(self ) -> None:
        pass

    @abc.abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        raise NotImplementedError("Please Implement fetch_data")

class ResultData(Data):
    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'result'
    @classmethod
    def PATH_SUFFIX(cls)->str:
        return 'race'
    
    @classmethod
    def ID_PREFIX(cls)->str:
        return cls.PATH_SUFFIX()
    
    @classmethod
    def COL_TYPES(cls) -> dict:
        return {'type'   :str,
                'strike' :str,
                'return' :'int64',
        }
    
    @classmethod
    def COL_SIZE(cls) -> dict:
        return {
            'strike':24
        }
    
    def append_race_id(self, df:pd.DataFrame) -> pd.DataFrame:
        df['race_id'] = self.id
        return df

    def fetch_data(self) -> pd.DataFrame:
        df=self.tables[1]
        df.columns=['type','strike','return','popularity']
        df=df.drop('popularity', axis=1)
        df['return'] = pd.to_numeric(df['return'].apply(lambda x: "".join(re.findall(r'\d+', x))) , errors='coerce')       
        return self.append_race_id(df)

class RaceData(Data):

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'race'
    
    @classmethod
    def COL_SIZE(cls) -> dict:
        return {'horse_name': 100,
                'field_type': 24,
                'direction' : 24,
                'ground_condition':24,
                'weather':12,
                'date':24,
                'start_time':36}

    @classmethod
    def COL_TYPES(cls) -> dict:
        return {'rank'              :'int64',
                'lane_num'          :'int64',
                'horse_num'         :'int64',
                'horse_name'        :str,
                'load_weight'       :'float64',
                'jockey_name'       :'object',
                'time'              :'object',
                'rank_first_odd'    :'float64',
                'popularity'        :'int64',
                'sex'               :'object',
                'age'               :'int64',
                'horse_weight'      :'int64',
                'horse_weight_diff' :'int64',
                'norm_horse_weight' :'float64',
                'norm_load_weight'  :'float64',
                'race_id'           :'str',
                'distance'          :'str',
                'is_hindrance'      :'int64',
                'field_type'        :str,
                'direction'         :str,
                'ground_condition'  :str, 
                'weather'           :str, 
                'date'              :str, 
                'start_time'        :str, 
                'horse_id'          :str, 
                'jockey_id'         :str}
    
    def fetch_data(self) -> pd.DataFrame:
        df = self.fetch_result()
        df = pd.merge(df, self.fetch_race_meta(), on='race_id')
        df = pd.merge(df, self.fetch_horse_id(), on='horse_name')
        df = pd.merge(df, self.fetch_jockey_id(), on='jockey_name')
        return df 

    def append_race_id(self, df:pd.DataFrame) -> pd.DataFrame:
        df['race_id'] = self.id
        return df

    def fetch_result(self):
        df = self.tables[0]
        df = df[~(df["着順"].astype(str).str.contains('\\D'))]
        df = deepcopy(df)
        df["着順"] = df["着順"].astype(int)
        
        df["sex"] = df["性齢"].map(lambda x: str(x)[0])
        df["age"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        df["horse_weight"] = df["馬体重"].str.split("(", expand=True)[0]
        df=df[~(df["horse_weight"].astype(str).str.contains('\\D'))]
        df["horse_weight"] = df["horse_weight"].astype(int)
        df["horse_weight_diff"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

        df["単勝"] = df["単勝"].astype(float)

        df.drop(["着差", "調教師", "性齢", "馬体重"], axis=1, inplace=True)
        
        df["タイム"] = df["タイム"]
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
        def decompose(val):
            val = val.replace('m', '')
            res= dict()
            res['distance'] = re.findall(r'\d+', val)[0]
            field=val.replace(res['distance'], "")
            values = re.findall(r'\w', field)
            def get_field_type(input, options):
                fields = []
                for ch in input:
                    if ch in options:
                        fields.append(ch)
                return '/'.join(fields)
            if values[0] =="障":
                res['is_hindrance'] = 1
                field_type = values[1:]
                res['field_type'] = get_field_type(field_type, ["芝", "ダ"])
            else:
                res['is_hindrance'] = 0
                res['field_type'] = get_field_type(values, ["芝", "ダ"])
            res['direction'] = get_field_type(values, ["左", "右", '内', '外'])
            return res
        if isinstance(tag, bs4.element.Tag):
            infos=re.findall(r'\w+',
                            tag.find_all('p')[0].text + tag.find_all('p')[1].text)
            infos += re.findall(r'\w+:\w+', tag.find_all('p')[0].text)
            idx = [i for i,x in enumerate(infos) if re.search( r'\d+m', x)][0]
            res = decompose(''.join(infos[:idx+1]))
            ground_condition = []
            for info in infos:
                if info in ['良','稍重', '重', '不良']:
                    ground_condition.append(info)
                if info in ['曇', '晴', '雨', '小雪', '雪','小雨']:
                    res['weather']= info
                if '年' in info and '月' in info:
                    res['date']= info
                if ':' in info:
                    res['start_time']= info
                res['ground_condition'] = '/'.join(ground_condition)
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
        if not isinstance(df, pd.DataFrame):
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

    def process_data(self) -> None:
        self.data['time'] = self.data['time'].apply(lambda x: racetimestr2timedelta(x))
        self.data['date']= self.data['date'].apply(lambda x: dt.datetime.strptime(x, '%Y年%m月%d日').date())
        self.data['start_time']= self.data['start_time'].apply(lambda x:dt.datetime.strptime(x, '%H:%M').time())


class HorseData(Data):

    def __init__(self,*, main_only=True, **kwargs) -> None:
        super().__init__(**kwargs)
        if main_only:
            self.filter_to_main_races()

    @classmethod
    def DATA_TYPE(cls) -> str:
        return 'horse'

    @classmethod
    def COL_SIZE(cls) -> dict:
        return {'field_type': 24,
                'weather':12,
                'host':12,
                'date':24,
                'time':36}

    @classmethod
    def COL_TYPES(cls) -> dict:
        return {'date'              :str, 
                'host'              :str,
                'rank'              :'int64',
                'round'             :'int64',
                'load_weight'       :'float64',
                'jockey_name'       :'object',
                'time'              :'object',
                'popularity'        :'int64',
                'distance'          :'str',
                'field_type'        :str,
                'weather'           :str,
                'condition'         :str,
                'bonus'             :'float64',
                'progress'          :str}

    def fetch_data(self) -> pd.DataFrame:
        '''
        ['日付', '開催', '天気', 'R', '頭数', '枠番', '馬番', 'オッズ', '人気',
         '着順', '騎手', '斤量', '距離', '馬場', 'タイム', '着差', '通過', 'ペース',
         '上り', '馬体重', '厩舎ｺﾒﾝﾄ', '備考', '勝ち馬(2着馬)', '賞金']
        '''
        data = self.fetch_result().loc[:,['日付', '開催', '天気', 'R', '人気',
        '着順', '騎手', '斤量', '距離', '馬場', 'タイム', '通過', '賞金']]
        data = data[~(data["着順"].astype(str).str.contains('\\D'))]
        col_names= ['date', 'host', 'weather', 'round', 'popularity', 'rank']
        col_names+=['jockey_name', 'load_weight', 'distance', 'condition']
        col_names+=['time', 'progress', 'bonus']
        data.columns = col_names
        data['field_type'] = data['distance'].apply(lambda x: x.replace(re.findall(r'\d+', x)[0], ''))
        data['distance'] = data['distance'].apply(lambda x: re.findall(r'\d+', x)[0])
        data['bonus']=data['bonus'].fillna(0)
        data['host'] = data['host'].apply(lambda x:''.join(filter(lambda x: x.isalpha(), x)))
        data['horse_id']=self.id
        return data[data['rank'].notna() & data['rank'].apply(lambda x: True if isinstance(x, numbers.Number) else x.isnumeric())]

    def fetch_result(self):
        index = [i for i,x in enumerate(self.tables) if '着順' in x.columns][0]
        return self.tables[index]

    def data_as_of_date(self, asofdate:dt.date):
        return self.data[pd.to_datetime(self.data['date'], format='%Y/%m/%d').dt.date<=asofdate]  # type: ignore

    def process_data(self) -> None:
        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y/%m/%d')
    
    def filter_to_main_races(self) -> None:
        main_races =  ['東京', '中山', '札幌', '函館', '小倉', '中京', '新潟', '阪神', '京都', '福島']
        self.data = self.data[self.data['host'].apply(lambda x: x in main_races)]

    