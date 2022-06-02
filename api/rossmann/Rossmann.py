import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime


class Rossmann(object):
    def __init__(self):
#     self.home_path = 'path'
#     self.model_pipeline  = pickle.load(open('path', 'rb'))
        return None

    def data_cleaning(self, data1 ):
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                'Promo2SinceYear', 'PromoInterval']
        
        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map(snakecase,cols_old))
        data1.columns = cols_new

        data1.date = pd.to_datetime(data1.date)

        # Fill NA
        data1['competition_distance'] = data1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)
        data1['competition_open_since_month'] = data1.apply(lambda x: x['date'].month 
                                                        if math.isnan(x['competition_open_since_month']) 
                                                        else x['competition_open_since_month'], axis=1)

        data1['competition_open_since_year'] = data1.apply(lambda x: x['date'].year 
                                                      if math.isnan(x['competition_open_since_year']) 
                                                      else x['competition_open_since_year'], 
                                                      axis=1)
                
        data1['promo2_since_week'] = data1.apply(lambda x: x['date'].week 
                                            if math.isnan(x['promo2_since_week']) 
                                            else x['promo2_since_week'], 
                                            axis=1)

        data1['promo2_since_year'] = data1.apply(lambda x: x['date'].year 
                                            if math.isnan(x['promo2_since_year']) 
                                            else x['promo2_since_year'], 
                                            axis=1)

        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 
                4:'Apr', 5:'May', 6:'Jun', 
                7:'Jul', 8:'Aug', 9:'Sept', 
                10:'Oct', 11:'Nov', 12:'Dec'}

        data1['promo_interval'].fillna(0, inplace=True)
        data1['month_map'] = data1['date'].dt.month.map(month_map)

        data1['is_promo'] = data1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

    ## Change Types
        data1['competition_open_since_month'] = data1['competition_open_since_month'].astype(int)
        data1['competition_open_since_year'] = data1['competition_open_since_year'].astype(int)
        data1['promo2_since_week'] = data1['promo2_since_week'].astype(int)
        data1['promo2_since_year'] = data1['promo2_since_year'].astype(int)
        data1['open'] = data1['open'].astype(int)

        return data1

    def feature_engineering(self, data2):
        # year
        data2['year'] = data2['date'].dt.year

        # month
        data2['month'] = data2['date'].dt.month

        # day
        data2['day'] = data2['date'].dt.day

        # year week
        data2['week_of_year'] = data2['date'].dt.weekofyear

        # week of year
        data2['year_week'] = data2['date'].dt.strftime('%Y-%W')

        # competition since
        data2['competition_since'] = data2.apply(lambda x: datetime.datetime(year =x['competition_open_since_year'] , 
                                                                        month =x['competition_open_since_month'], 
                                                                        day=1),axis=1)
        # manter em meses - diferença de meses
        data2['competition_time_month'] = ((data2['date'] - data2['competition_since'].values.astype('datetime64[ns]'))/30).apply(lambda x: x.days).astype(int)

        # join das duas colunas. 
        data2['promo_since'] = data2['promo2_since_year'].astype(str) + '-' + data2['promo2_since_week'].astype(str)

        data2['promo_since'] = data2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

        data2['promo_time_week'] = ((data2['date'] - data2['promo_since'])/7).apply(lambda x: x.days).astype(int)
        # assortment
        data2['assortment'] = data2['assortment'].apply(lambda x: 'basic' if x == 'a' 
                                                    else 'extra' if x == 'b' 
                                                    else 'extended')

        # state holiday
        data2['state_holiday'] = data2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' 
                                                    else 'easter_holiday' if x == 'b' 
                                                    else 'christmas' if x == 'c' else 'regular_day')
    
        # filtragem de linhas
        data2 = data2[(data2['open'] != 0)]  

        # exclui a open tbm porque ela só tem 1 depois que filtra
        # outras são retiradas por serem só auxiliares
        cols_drop = ['open', 'promo_interval', 'month_map']
        data2 = data2.drop(cols_drop, axis=1)

        return data2

    def data_prep(self, data5):
        # sincos
  
        data5['day_of_week_sin'] = data5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
        data5['day_of_week_cos'] = data5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))

        # month
        data5['month_sin'] = data5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        data5['month_cos'] = data5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))

        # day
        data5['day_sin'] = data5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
        data5['day_cos'] = data5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))

        # week of year
        data5['week_of_year_sin'] = data5['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/52)))
        data5['week_of_year_cos'] = data5['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/52)))

        # data5['sales'] = np.log1p(data5['sales']) ????

        boruta_selected = ['store','promo','store_type', 'assortment','competition_distance','competition_open_since_month',
                'competition_open_since_year','promo2','promo2_since_week','promo2_since_year','competition_time_month',
                'promo_time_week','day_of_week_sin','day_of_week_cos','month_sin','month_cos','day_sin','day_cos',
                'week_of_year_sin','week_of_year_cos']

        return data5[boruta_selected]
    
  # pipeline e fit 
    def get_prediction(self,model, original_data, test_data):
        
        pred = model.predict(test_data)
        # join pred into original data
        original_data['prediction'] = np.expm1(pred)
    
        return original_data.to_json(orient='records', date_format='iso')

