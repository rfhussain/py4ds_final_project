import pandas as pd
import numpy as np
import gc
import os

from tqdm import tqdm_notebook
from datetime import datetime
import calendar
from itertools import product
#from googletrans import Translator 'doesn't work after sometime'
from translate import Translator
from sklearn.preprocessing import LabelEncoder
import datetime as DT
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz

from collections import Counter
import re
from operator import itemgetter


import warnings
warnings.filterwarnings("ignore")

class SalesUtils():

    def __init__(self, submission_path):
        super().__init__()
        self.__index_cols = ['shop_id', 'item_id', 'date_block_num']
        self.__today = str(DT.date.today())
        self.__submission_path = submission_path
        self.__input_path = '../data/'

    def parse_shop_names(self, df):

        '''
        this function will extract the city name from the shops
        will also add the city_id by applying the label encoding to it.

        parameters
        ----------------    
        df : data frame object for shops
        '''

        translated_shops = ['Yakutsk Ordzhonikidze, 56 franc', 'Yakutsk shopping center Central fran', 'Adygea TC "Mega"', 'Balashikha shopping and entertainment complex "October-Kinomir"',
        'Volga Shopping Center Volga Mall ', 'Vologda shopping center "Marmalade"', 'Voronezh (Plekhanovskaya, 13)', 'Voronezh SEC "Maximir"', 'Voronezh SEC City-Park Grad',
        'Outbound Trading', ' Zhukovsky st. Chkalova 39m? ', ' Zhukovsky st. Chkalova 39m² ', 'Online emergency store', 'Kazan shopping center Bechetle ',
        'Kazan Shopping Center Park House II', 'Kaluga shopping and entertainment center "XXI Century"', 'Kolomna shopping center "Rio"', 'Krasnoyarsk shopping center Rise of the Plaza ',
        'Krasnoyarsk shopping center "June"', 'Kursk shopping center Pushkinsky ', 'Moscow Sale', 'Moscow MTRC "Afi Mall"', 'Moscow Shop C21', 'Moscow TC "Budenovsky" (Pav.A2)',
        'Moscow TC "Budenovsky" (Pavilion K7)', 'Moscow TRC Atrium', 'Moscow shopping center "Areal" (Belyaevo)', 'Moscow shopping center "MEGA Belaya Dacha II"', 
        'Moscow shopping center "MEGA Teply Stan" II', 'Moscow Shopping Center "New Century" (Novokosino)', 'Moscow shopping center "Perlovsky"', 'Moscow shopping center "Semenovsky"',
        'Moscow Shopping Center "Silver House"', 'Mytishchi TRK "XL-3"', 'N. Novgorod SEC "RIO"', 'N. Novgorod shopping and entertainment center “Fantasy”', 
        'Novosibirsk shopping center "Gallery Novosibirsk"', 'Novosibirsk shopping center "Mega"','Omsk shopping center Mega ', 
        'RostovNaDon shopping and entertainment complex "Megacenter Horizon"', 'RostovNaDon shopping and entertainment complex Megacenter Horizon Island',
        'RostovNaDon shopping center "Mega"', 'St.Petersburg TC "Nevsky Center"', 'St.Petersburg shopping center Sennaya ', 'Samara shopping center Melody ',
        'Samara Shopping Center Park House', 'Sergiev Posad shopping center "7Я"', 'Surgut SEC "City Mall"', 'Tomsk shopping center "Emerald City"', 'Tyumen SEC "Crystal"',
        'Tyumen shopping center "Goodwin"', 'Tyumen shopping center "Green Beach"', 'Ufa shopping center "Central"', 'Ufa shopping center Family 2', 'Khimki shopping center Mega ',
        'Digital warehouse 1C-Online', 'Chekhov SEC "Carnival"', 'Yakutsk Ordzhonikidze, 56', 'Yakutsk shopping center "Central"', 'Yaroslavl Shopping Center Altair']

        #translator= Translator(to_lang="en", from_lang="ru") #using the pypy translator

        #first two rows have ! mark in the shop_name, removing it
        df.at[0,'shop_name'] = str(df.loc[0]['shop_name']).replace('!','')
        df.at[1,'shop_name'] = str(df.loc[1]['shop_name']).replace('!','')

        #getting the shop name in english
        #shop_name_en = [translator.translate(shop_name) for shop_name in tqdm_notebook(df['shop_name'].tolist())] 
        #df['shop_name_en'] = shop_name_en

        df['shop_name_en'] = translated_shops

        #getting the city name from the shop_name
        df['city_name'] = df['shop_name_en'].apply(lambda x: x.strip().split(' ')[0])

        #label encoding for city
        le = LabelEncoder()
        df['city_id'] = le.fit_transform(df['city_name'].tolist())

        return df


    def parse_item_categories(self, df):

        '''
        below list is translated from google translate.         
        '''
        item_cat_en = ['PC - Headsets / Headphones','Accessories - PS2','Accessories - PS3','Accessories - PS4','Accessories - PSP','Accessories - PSVita',
        'Accessories - XBOX 360','Accessories - XBOX ONE','Tickets (Digit)','Delivery of goods','Game consoles - PS2','Game consoles - PS3','Game consoles - PS4',
        'Game Consoles - PSP','Game consoles - PSVita','Game consoles - XBOX 360','Game consoles - XBOX ONE','Game consoles - Other','Games - PS2','Games - PS3',
        'Games - PS4','Games - PSP','Games - PSVita','Games - XBOX 360','Games - XBOX ONE','Games - Accessories for games','Android Games - The Number','MAC Games - Digit',
        'PC Games - Additional Editions','PC Games - Collectors Editions','PC Games - Standard Editions','PC Games - The Number','Payment cards (Cinema, Music, Games)',
        'Payment Cards - Live!','Payment Cards - Live! (Numeral)','Payment Cards - PSN','Payment Cards - Windows (Digit)','Cinema - Blu-Ray','Cinema - Blu-Ray 3D',
        'Cinema - Blu-Ray 4K','Cinema - DVD','Cinema - Collectible','Books - Artbooks, Encyclopedias','Books - Audiobooks','Books - Audiobooks (Digit)','Books - Audiobooks 1C',
        'Books - Business Literature','Books - Comics, Manga','Books - Computer Literature','Books - Methodological materials 1C','Books - Postcards','Books - Cognitive literature',
        'Books - Guides','Books - Fiction','Books - The Number','Music - Local Production CD','Music - Brand-name CD','Music - MP3','Music - Vinyl','Music - Music Video',
        'Music - Gift Editions','Gifts - Attributes','Gifts - Gadgets, Robots, Sports','Gifts - Soft Toys','Gifts - Board Games','Gifts - Board games (compact)',
        'Gifts - Postcards, stickers','Gifts - Development','Gifts - Certificates, services','Gifts - Souvenirs','Gifts - Souvenirs (in bulk)','Gifts - Bags, Albums, Mousepads',
        'Gifts - Figures','Programs - 1C: Enterprise 8','Programs - MAC (Digit)','Programs - For home and office','Programs - For home and office (Digital)',
        'Programs - Educational','Programs - Educational (Digital)','Service','Service - Tickets','Clean carriers (spire)','Blank media (piece)','Batteries']

        df['item_cat_en'] = item_cat_en

        #updating the parent category
        df['parent_cat'] = ''

        df.at[0 ,'parent_cat'] = 'headphones'
        df.at[8 ,'parent_cat'] = 'tickets'
        df.at[9 ,'parent_cat'] = 'delivery of goods'
        df.at[25,'parent_cat'] = 'accessories games'
        df.at[26,'parent_cat'] = 'android games'
        df.at[27,'parent_cat'] = 'mac games'
        df.at[81,'parent_cat'] = 'carriers'
        df.at[82,'parent_cat'] = 'blank media'
        df.at[83,'parent_cat'] = 'batteries'

        for i in range(1 , 8): df.at[i,'parent_cat'] = 'accessories games'
        for i in range(10,18): df.at[i,'parent_cat'] = 'game consoles'
        for i in range(18,25): df.at[i,'parent_cat'] = 'games'
        for i in range(28,32): df.at[i,'parent_cat'] = 'pc games'
        for i in range(32,37): df.at[i,'parent_cat'] = 'payment cards'
        for i in range(37,42): df.at[i,'parent_cat'] = 'cinema'
        for i in range(42,55): df.at[i,'parent_cat'] = 'books'
        for i in range(55,61): df.at[i,'parent_cat'] = 'music'
        for i in range(61,73): df.at[i,'parent_cat'] = 'gifts'
        for i in range(73,79): df.at[i,'parent_cat'] = 'programs'
        for i in range(79,81): df.at[i,'parent_cat'] = 'service'

        le = LabelEncoder()
        df['parent_cat_id'] = le.fit_transform(df['parent_cat'])

        return df        

    def get_days_count(self):
        '''
        this function will return the number of sundays, saturdays, and num_days
        that will be based on the date_block_num
        '''
        date_block_counter = 0
        dbn = []
        num_sat = []
        num_sun = []
        num_days = []

        for year in [2013,2014,2015]:
            for month in range(1,13):
                
                #addup the date block num
                dbn.append(date_block_counter)
                
                #count saturday
                num_sat.append(len([1 for i in calendar.monthcalendar(year,month) if i[calendar.SATURDAY] != 0]))
                
                #count sunday
                num_sun.append(len([1 for i in calendar.monthcalendar(year,month) if i[calendar.SUNDAY] != 0]))
                
                #count total days
                num_days.append(calendar.monthrange(year,month)[1])
                
                if date_block_counter == 34: break
                date_block_counter+=1 
                

        week_day_count = pd.DataFrame(np.c_[dbn,num_sat,num_sun,num_days], columns=['date_block_num','num_sat','num_sun','num_days'])

        return week_day_count


    def __name_correction(self,x):
        x = x.lower()
        x = x.partition('[')[0]
        x = x.partition('(')[0]
        x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
        re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
        x = x.strip()
        return x

    def parse_items(self,df_items):
        # creating two additional features
        # note: 
        # replace function if second argument passed as 1 i.e.############ replace('[',1)
        # then it will only split in 2 values, although complete split may have been 5 pieces
        df_items['name_1'], df_items['name_2'] = df_items['item_name'].str.split('[', 1).str # this case name_3 will be null
        df_items['name_1'], df_items['name_3'] = df_items['item_name'].str.split('(', 1).str # this case name_2 will be null

        # removing unwanted characters from name_2 and name_3
        df_items['name_2'] = df_items['name_2'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
        df_items['name_3'] = df_items['name_3'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
        df_items = df_items.fillna('0')

        ######################################################################
        ############ TEMPORARY FEATURE SET, BUT NOT USED          ############
        ######################################################################
        df1 = Counter(' '.join(df_items['name_2'].values).split(' ')).items()
        df1 = sorted(df1, key=itemgetter(1))
        df1 = pd.DataFrame(df1, columns=['feature', 'count'])
        df1.fillna(0)
        df1 = df1[(df1['feature'].str.len() > 1) & (df1['count'] > 200)]

        df2 = Counter(' '.join(df_items['name_3'].values).split(' ')).items()
        df2 = sorted(df2, key=itemgetter(1))
        df2 = pd.DataFrame(df2, columns=['feature', 'count'])
        df2.fillna(0)
        df2 = df2[(df2['feature'].str.len() > 1) & (df2['count'] > 200)]

        item_feature_set = pd.concat([df1,df2])
        item_feature_set = item_feature_set.drop_duplicates(subset=['feature']).reset_index(drop=True)

        ######################################################################

        # correcting the item name, and removing unwanted characters, empty spaces etc
        df_items['item_name'] = df_items['item_name'] .apply(lambda x: self.__name_correction(x))

        # removing the preceeding white space from name_2
        df_items.name_2 = df_items.name_2.apply(lambda x: x[:-1] if x!='0' else '0')

        ######################################################################
        ############ TEMP FEATURE TYPE, IT WILL GO TO NAME_2      ############
        ######################################################################
        df_items['type'] = df_items.name_2.apply(lambda x: x[0:8] if x.split(' ')[0] == 'xbox' else x.split(' ')[0])
        df_items.loc[(df_items.type == 'x360') | (df_items.type == 'xbox360'), 'type'] = 'xbox 360'
        df_items.loc[df_items.type == '', 'type'] = 'mac'
        df_items.type = df_items.type.apply(lambda x: x.replace(' ',''))

        df_items.loc[(df_items.type == 'pc') | (df_items.type == 'pс') | (df_items.type == 'рс'), 'type'] = 'pc'
        df_items.loc[(df_items.type == 'рs3'), 'type'] = 'ps3'

        #the sum of all item_category_ids under a particular type should not be less than 200
        group_sum = df_items.groupby('type', as_index=False).sum()
        to_del_types = group_sum.loc[group_sum.item_category_id < 200].type.tolist() 

        # apply it to name_2 and delete type field
        df_items.name_2 = df_items.type.apply(lambda x: 'etc' if x in to_del_types else x)

        # remove the type
        df_items = df_items.drop(['type'], axis=1)

        # apply label encoder, and remove the item_name, and name_1
        df_items['name_2'] = LabelEncoder().fit_transform(df_items['name_2'])
        df_items['name_3'] = LabelEncoder().fit_transform(df_items['name_3'])
        df_items.drop(['item_name', 'name_1'], axis=1, inplace=True)

        # return the items data frame
        return df_items


    def add_date_attributes(self, df):
        dfmain = df
        dfmain['day'] = dfmain['date'].dt.day
        dfmain['month'] = dfmain['date'].dt.month
        dfmain['weekday'] = dfmain['date'].dt.day_name()
        dfmain['weekdayno'] = dfmain['date'].dt.dayofweek
        dfmain['year'] = dfmain['date'].dt.year
        return dfmain

    def merge_items_sales_n_shops(self,df):

        # get the data-sets
        df_items           = pd.read_csv(os.path.join(self.__input_path, 'items.csv'))
        df_item_categories = pd.read_csv(os.path.join(self.__input_path, 'item_categories.csv'))
        df_shops           = pd.read_csv(os.path.join(self.__input_path, 'shops.csv'))

        ############    ITEMS      ##############
        # parsing the item categories (i.e. adding the parent category on top of the item_category)
        df_item_categories = self.parse_item_categories(df_item_categories)

        # parsing the items data set
        df_items = self.parse_items(df_items)

        # merging the items with item categories
        df_items = df_items.merge(df_item_categories, how='inner', on=['item_category_id'])

        ############    SHOPS      ##############
        # Adding the city_id and city_name to the shops
        df_shops = self.parse_shop_names(df_shops)

        # merging the sales with the items
        df = df.merge(df_items, how='inner')

        # merging the sales with the shops 
        df = df.merge(df_shops[['shop_id','city_id','city_name']].drop_duplicates(), how='inner')

        return df
    
    def get_matrix_by_block(self, df):
        '''
        this function will convert the dataframe to matrix for shop_id/item_id with the constant of date_block_num
        In other words, it will be a matrix of each month vertically stacked together
        
        input   : dataframe
        output  : dataframe
        '''
        
        grid = []

        #loop through each block in the data
        for block_num in tqdm_notebook(df['date_block_num'].unique()):
            
            cur_shops = df[df['date_block_num']==block_num]['shop_id'].unique()
            cur_items = df[df['date_block_num']==block_num]['item_id'].unique()
            cur_month = df[df['date_block_num']==block_num]['month'].unique()
            
            grid.append(np.array(list(product(*[cur_shops,cur_items,[block_num],[cur_month]])),dtype='int32'))
            
        #turn the grid into pandas dataframe
        grid = pd.DataFrame(np.vstack(grid), columns=['shop_id', 'item_id', 'date_block_num','month'], dtype=np.int32)    
        

        ############ SUM OF ITEM_CNT_DAY ..> TARGET ###########
        #get the aggregated value for (shop_id, item_id, date_block_num)
        gb = df.groupby(self.__index_cols, as_index=False).agg({'item_cnt_day':'sum'})
        gb.rename(columns={'item_cnt_day':'target'}, inplace=True)
        #joining the agregated data to the grid
        df_all_data = pd.merge(grid,gb, how='left', on=self.__index_cols).fillna(0)

        ############ SUM OF REVENUE ###########
        #get the aggregated revenue for (shop_id, item_id, date_block_num)
        gb = df.groupby(self.__index_cols, as_index=False).agg({'revenue':'sum'})        
        #joining the agregated data to the grid
        df_all_data = pd.merge(df_all_data,gb, how='left', on=self.__index_cols).fillna(0)

        
        # Same as above but with shop-month aggregates
        #gb = df.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum'}})
        gb = df.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':'sum'})
        gb.rename(columns={'item_cnt_day':'target_shop'}, inplace=True)
        #merge with main df
        df_all_data = pd.merge(df_all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

        gb = df.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'revenue':'sum'})
        gb.rename(columns={'revenue':'revenue_shop'}, inplace=True)
        #merge with main df
        df_all_data = pd.merge(df_all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)


        # Same as above but with item-month aggregates
        #gb = sales.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum'}})
        gb = df.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':'sum'})
        gb.rename(columns={'item_cnt_day':'target_item'}, inplace=True)    
        #merge with main df
        df_all_data = pd.merge(df_all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)


        gb = df.groupby(['item_id', 'date_block_num'],as_index=False).agg({'revenue':'sum'})
        gb.rename(columns={'revenue':'revenue_item'}, inplace=True)    
        #merge with main df
        df_all_data = pd.merge(df_all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)
        

        del gb
        del grid

        #sorting values
        df_all_data.sort_values(['date_block_num','shop_id','item_id'], inplace=True)
        
        return df_all_data    

    def add_mean_features(self, dfmain,features_to_add):

        if 1 in features_to_add:
            dfmain = self.add_expanding_mean(dfmain,['shop_id','item_id'])
        
        if 2 in features_to_add:
            dfmain = self.add_shop_item_target_mean(dfmain)

        if 3 in features_to_add:
            dfmain = self.add_item_id_target_mean(dfmain)

        if 4 in features_to_add:
            dfmain = self.add_month_mean_target(dfmain)

        if 5 in features_to_add:
            dfmain = self.add_parent_cat_mean_target(dfmain)

        if 6 in features_to_add:
            dfmain = self.add_item_category_mean_target(dfmain)

        if 7 in features_to_add:
            dfmain = self.add_shop_id_mean_target(dfmain)

        if 8 in features_to_add:
            dfmain = self.add_city_id_mean_target(dfmain)

        if 9 in features_to_add:
            dfmain = self.add_shop_city_target_mean(dfmain)

        if 10 in features_to_add:
            dfmain = self.add_date_block_target_mean(dfmain)

        return dfmain

    def add_expanding_mean(self,df, cols_to_add):
        
        '''
        This function will return the expanding mean 
        example as follows:

        city        target  cumsum  sumcount    city_expanding_mean
        -----------------------------------------------------------
        karachi     3       3       1           3.00
        karachi     2       5       2           2.50
        karachi     7       12      3           4.00
        karachi     -1      11      4           2.75
        karachi     4       5       5           3.00

        '''
        for col in cols_to_add:

            global_mean = df['target'].mean()

            cum_sum = df.groupby(col).target.cumsum() - df['target']
            sum_cnt = df.groupby(col).cumcount()

            #adding the expanding mean
            df[col + '_exp_y_mean'] = cum_sum/sum_cnt

            # filling the null
            df[col + '_exp_y_mean'].fillna(global_mean, inplace=True)
        
        return df

    def add_shop_item_target_mean(self,dfmain):
        '''
        This function will group by the shop_id and item_id and add the mean         
        '''
        df_shop_item_mean = pd.DataFrame(dfmain.groupby(['shop_id','item_id','date_block_num'], as_index=False).target.mean())
        df_shop_item_mean.rename(columns={'target':'shop_item_db_mean'}, inplace=True)
        dfmain = dfmain.merge(df_shop_item_mean, how='left')
        return dfmain
    
    def add_shop_city_target_mean(self,dfmain):
        '''
        This function will group by the shop_id and city_id and add the mean         
        '''
        df_shop_city_mean = pd.DataFrame(dfmain.groupby(['shop_id','city_id'], as_index=False).target.mean())
        df_shop_city_mean.rename(columns={'target':'shop_city_mean'}, inplace=True)
        dfmain = dfmain.merge(df_shop_city_mean, how='left')
        return dfmain

    def add_item_id_target_mean(self, dfmain):
        '''
        this function will calculate the item_id target_mean
        '''
        means = dfmain.groupby(['item_id','date_block_num'], as_index=False).target.mean()
        means.rename(columns={'target':'item_id_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means, how='left')
        return dfmain

    def add_month_mean_target(self, dfmain):
        df_2013_2014 = dfmain # dfmain[dfmain['date_block_num'] < 24]
        means = df_2013_2014.groupby('month').target.mean()
        dfmain['month_mean_target'] = dfmain['month'].map(means)
        return dfmain

    def add_parent_cat_mean_target(self, dfmain):
        # adding the means with out date block number    
        tmpmeans = dfmain.groupby('parent_cat_id').target.mean()
        dfmain['parent_cat_id_mean_target'] = dfmain['parent_cat_id'].map(tmpmeans)

        # calculating the mean from sales_train
        means_parent_cat = dfmain.groupby(['parent_cat_id','date_block_num']).target.mean() 
        # renaming the column and merging
        means_parent_cat.rename(columns={'target':'parent_cat_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means_parent_cat, how ='left')
        return dfmain

    def add_item_category_mean_target(self, dfmain):

        # adding the means with out date block number    
        tmpmeans = dfmain.groupby('item_category_id').target.mean()
        dfmain['item_cat_mean_target'] = dfmain['item_category_id'].map(tmpmeans)

        means_cat = dfmain.groupby(['item_category_id','date_block_num'], as_index=False).target.mean()        
        means_cat.rename(columns={'target':'item_cat_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means_cat, how ='left')
        return dfmain
    
    def add_shop_id_mean_target(self, dfmain):

        # adding the means with out date block number    
        tmpmeans = dfmain.groupby('shop_id').target.mean()
        dfmain['shop_id_mean_target'] = dfmain['shop_id'].map(tmpmeans)

        means_shops = dfmain.groupby(['shop_id','date_block_num'], as_index=False).target.mean()
        means_shops.rename(columns={'target':'shop_id_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means_shops, how ='left')
        return dfmain

    def add_city_id_mean_target(self, dfmain):
        means_city = dfmain.groupby(['city_id','date_block_num'], as_index=False).target.mean()        
        means_city.rename(columns={'target':'city_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means_city, how ='left')
        return dfmain

    def add_date_block_target_mean(self,dfmain):
        db_mean = dfmain.groupby('date_block_num').target.mean()
        dfmain['db_target_mean'] = dfmain['date_block_num'].map(db_mean)
        return dfmain
    
    def add_lags(self, df, shift_range, exception_cols):
        cols_to_rename = list(df.columns.difference(self.__index_cols + exception_cols)) 

        for month_shift in tqdm_notebook(shift_range):
            train_shift = df[self.__index_cols + cols_to_rename].copy()

            train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

            foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
            train_shift = train_shift.rename(columns=foo)

            df = pd.merge(df, train_shift, on=self.__index_cols, how='left').fillna(0)

        return df

    def save_submission(self, model_name, df):
        '''
        The function will construct the file submission name, and save it to the specified path for submissions
        '''
        file_name =  'sub_' + model_name
        save_path = self.__submission_path + '\\' + file_name + '_' + str(self.__today) + '.csv'
        df.to_csv(save_path, index=False)
        print('Submission file saved as ' + save_path)




