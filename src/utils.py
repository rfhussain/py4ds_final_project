import pandas as pd
import numpy as np
import gc
import os

from tqdm import tqdm_notebook
from datetime import datetime
from itertools import product
#from googletrans import Translator 'doesn't work after sometime'
from translate import Translator
from sklearn.preprocessing import LabelEncoder
import datetime as DT
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz


import warnings
warnings.filterwarnings("ignore")

class SalesUtils():

    def __init__(self, submission_path):
        super().__init__()
        self.__index_cols = ['shop_id', 'item_id', 'date_block_num']
        self.__today = str(DT.date.today())
        self.__submission_path = submission_path

    
    def merge_sales_n_shops(self,data_folder,df):

        # get the data-sets
        df_items           = pd.read_csv(os.path.join(data_folder, 'items.csv'))
        df_item_categories = pd.read_csv(os.path.join(data_folder, 'item_categories.csv'))
        df_shops           = pd.read_csv(os.path.join(data_folder, 'shops.csv'))

        ############    ITEMS      ##############
        # parsing the item categories (i.e. adding the parent category on top of the item_category)
        df_item_categories = self.parse_item_categories(df_item_categories)
        # merging the items with item categories
        df_items = df_items.merge(df_item_categories, how='inner', on=['item_category_id'])

        ############    SHOPS      ##############
        # Adding the city_id and city_name to the shops
        df_shops = self.parse_shop_names(df_shops)

        # merging the sales with the items
        df = df.merge(df_items[['item_id','item_category_id','parent_cat_id','parent_cat']].drop_duplicates(), how='inner')

        # merging the sales with the shops 
        df = df.merge(df_shops[['shop_id','city_id','city_name']].drop_duplicates(), how='inner')

        return df
    
    
    def plot_sales_by_x(self,x,year,df):
        if ((year is not None) & (int(year)>1900)):
            #print('yes')   
            df_tmp = df[df['year'] == year]
        else:
            print('no')
            df_tmp = df
            
        sold_by_x = pd.DataFrame(df_tmp.groupby([x])['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day'))
        
        print(type(sold_by_x))
        
        sns.set_context("paper", font_scale=1.1)
        fig , ax = plt.subplots(figsize=(15,7))
        sns.barplot(x=x,y='item_cnt_day', data=sold_by_x, dodge=False)
        
        #top 5 cities
        if (x=='city_id'):
            t5 = sold_by_x.sort_values(by='item_cnt_day',ascending=False)['city_id'].head(5).values
            print('top 5 cities are :', t5)
            
        del sold_by_x,df_tmp

    def plot_sales_by_city(self,year,df):
        df_tmp = df[df['year'] == year]
        sold_by_city = pd.DataFrame(df_tmp.groupby('city_id')['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day'))
        sns.set_context("paper", font_scale=1.1)
        fig , ax = plt.subplots(figsize=(15,7))
        sns.barplot(x=sold_by_city.weekday,y=sold_by_city.item_cnt_day, order=sold_by_city.weekday)
        del sold_by_city,df_tmp

    def plot_sales_by_weekday(self,year,df):
        df_tmp = df[df['year'] == year]
        sold_by_weekday = pd.DataFrame(df_tmp.groupby('weekday')['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day'))
        sns.set_context("paper", font_scale=1.1)
        fig , ax = plt.subplots(figsize=(15,7))
        sns.barplot(x=sold_by_weekday.weekday,y=sold_by_weekday.item_cnt_day, order=sold_by_weekday.weekday)
        del sold_by_weekday,df_tmp

    def plot_sales_by_day(self,year,month,df):
        try:
            df_tmp = df[(df['year'] == year) & (df['month'] == month)]
            sold_by_day = pd.DataFrame(df_tmp.groupby('day')['item_cnt_day'].sum().reset_index())
            holidays = np.zeros(len(sold_by_day))

            sns.set_context("paper", font_scale=1.1)
            fig , ax = plt.subplots(figsize=(15,7))

            t = df_tmp[df_tmp.is_holiday==1]['day'].unique()
            for i in t: holidays[i-1]=1 
            holidays = pd.DataFrame(holidays, columns=['is_holiday'])
            sold_by_day = sold_by_day.merge(holidays, left_index=True, right_index=True)
            sold_by_day['is_holiday'] = sold_by_day['is_holiday'].astype('int8')
            sold_by_day = sold_by_day.sort_values(by='item_cnt_day', ascending=False)

            del df_tmp

            str_title = 'Year: ' + str(year) + ' : Month: '  + str(month)
            
            sns.barplot(x=sold_by_day.day,y=sold_by_day.item_cnt_day, hue=sold_by_day.is_holiday, dodge=False).set_title(str_title)
            del sold_by_day
        except:
            print('')
        



    def downcast_dtypes(self, df):
            '''
                Changes column types in the dataframe: 
                        
                        `float64` type to `float32`
                        `int64`   type to `int32`
            '''
            
            # Select columns to downcast
            float_cols = [c for c in df if df[c].dtype == "float64"]
            int_cols =   [c for c in df if df[c].dtype == "int64"]
            
            # Downcast
            df[float_cols] = df[float_cols].astype(np.float32)
            df[int_cols]   = df[int_cols].astype(np.int32)
            
            return df

    #Data Manipulation
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
     
    def add_lags(self, df, shift_range, exception_cols):
        cols_to_rename = list(df.columns.difference(self.__index_cols + exception_cols)) 

        for month_shift in tqdm_notebook(shift_range):
            train_shift = df[self.__index_cols + cols_to_rename].copy()

            train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

            foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
            train_shift = train_shift.rename(columns=foo)

            df = pd.merge(df, train_shift, on=self.__index_cols, how='left').fillna(0)

        return df

    def add_date_attributes(self, df):
        dfmain = df
        dfmain['day'] = dfmain['date'].dt.day
        dfmain['month'] = dfmain['date'].dt.month
        dfmain['weekday'] = dfmain['date'].dt.day_name()
        dfmain['weekdayno'] = dfmain['date'].dt.dayofweek
        dfmain['year'] = dfmain['date'].dt.year
        return dfmain
    
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



    def parse_item_categories_old(self, df):
        '''
        This function will extract parent categories from the item categoreis
        Also, it will label encode the parent catgory, and make it part of the dataframe
        and will return the dataframe.
        '''
        #translator = Translator() This google translate doesn't work after sometime.
        translator= Translator(to_lang="en", from_lang="ru") #using the pypy translator

        splitter = ''
        try:
            #below coe is pypy translator, not google trans, but it doesn't translate the char '-' sometimes
            '''
            commented....
            Translating the category name, 
            item_cat_list_en = [translator.translate(cat_name) for cat_name in tqdm_notebook(df['item_category_name'].tolist())] 
            '''
            #below google translate code not working so commented
            '''
            commented....
            parent_cat_en = [i.text for i in translator.translate(parent_cat, dest='en')] 
            '''

            #commented... even the category_name_en is not needed
            #df['item_category_name_en'] = item_cat_list_en
            
            #Creating the parent category
            df['parent_cat'] = '' #initialize empty
            
            for i in tqdm_notebook(range(len(df))):
                
                if len(str(df.loc[i]['item_category_name'])) <= 0: 
                    splitter = 'Temp'
                else:
                    splitter = str(df.loc[i]['item_category_name']).split('-')

                df.at[i,'parent_cat_en'] = translator.translate(splitter[0].strip())
                df.at[i,'parent_cat'] = splitter[0].strip()

            #label encoding the parent_cat
            
            le = LabelEncoder()
            df['parent_cat_id'] = le.fit_transform(df['parent_cat'].tolist())

            #return the dataframe
            return df

        except Exception as e:
            print(str(e) + splitter)
    
    def add_mean_features(self, dfmain,features_to_add):

        if 1 in features_to_add:
            dfmain = self.add_expanding_mean(dfmain,['shop_id'])
        
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
        df_shop_item_mean = pd.DataFrame(dfmain.groupby(['shop_id','item_id'], as_index=False).target.mean())
        df_shop_item_mean = df_shop_item_mean.rename(columns={'target':'shop_item_mean'})
        dfmain = dfmain.merge(df_shop_item_mean, on=['shop_id','item_id'], how='left')
        return dfmain
    
    def add_shop_city_target_mean(self,dfmain):
        '''
        This function will group by the shop_id and city_id and add the mean         
        '''
        df_shop_item_mean = pd.DataFrame(dfmain.groupby(['shop_id','city_id'], as_index=False).target.mean())
        df_shop_item_mean = df_shop_item_mean.rename(columns={'target':'shop_city_mean'})
        dfmain = dfmain.merge(df_shop_item_mean, on=['shop_id','city_id'], how='left')
        return dfmain

    def add_item_id_target_mean(self, dfmain):
        '''
        this function will calculate the item_id target_mean
        '''
        means = dfmain.groupby('item_id').target.mean()
        dfmain['item_id_mean_target'] = dfmain['item_id'].map(means)

        return dfmain

    def add_month_mean_target(self, dfmain):
        df_2013_2014 = dfmain # dfmain[dfmain['date_block_num'] < 24]
        means = df_2013_2014.groupby('month').target.mean()
        dfmain['month_mean_target'] = dfmain['month'].map(means)
        return dfmain

    def add_parent_cat_mean_target(self, dfmain):
        #calculating the mean from sales_train
        means_parent_cat = dfmain.groupby('parent_cat_id').target.mean() 
        #attaching the mean to the matrix dataframe (dfmain)
        dfmain['parent_cat_mean_target'] = dfmain['parent_cat_id'].map(means_parent_cat) 
        return dfmain

    def add_item_category_mean_target(self, dfmain):
        means_cat = dfmain.groupby('item_category_id').target.mean()
        dfmain['item_cat_mean_target'] = dfmain['item_category_id'].map(means_cat)

        means_cat = dfmain.groupby(['item_category_id','date_block_num'], as_index=False).target.mean()        
        means_cat.rename(columns={'target':'item_cat_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means_cat, how ='left')

        return dfmain
    
    def add_shop_id_mean_target(self, dfmain):
        means_shops = dfmain.groupby('shop_id').target.mean()
        dfmain['shop_id_mean_target'] = dfmain['shop_id'].map(means_shops)

        means_shops = dfmain.groupby(['shop_id','date_block_num'], as_index=False).target.mean()
        means_shops.rename(columns={'target':'shop_id_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means_shops, how ='left')

        return dfmain

    def add_city_id_mean_target(self, dfmain):
        means_city = dfmain.groupby('city_id').target.mean()
        dfmain['city_id_mean_target'] = dfmain['city_id'].map(means_city)

        means_city = dfmain.groupby(['city_id','date_block_num'], as_index=False).target.mean()        
        means_city.rename(columns={'target':'city_db_mean_target'}, inplace=True)
        dfmain = dfmain.merge(means_city, how ='left')
        
        return dfmain


    #This function is not being used at the moment
    def add_mean_attributes(self, dfmain):
        '''
            Mean target attribute for month
            Mean target attribute for ??

            Note: 
            On Experimental basis, all the means will be calculated on df (df_sales) the original dataframe
            The second try could be the mean calculation based on the matrix dataframe (but that will still not include, parent_cat_mean_target)

        '''
        '''
        for month:
        important thing about month is that, since November, December 2015 isn't available
        This is going to affect big time.
        I prefer to calculate the mean for month by omitting the year 2015 from the dataframe.

        '''
        df_2013_2014 = dfmain[dfmain['date_block_num'] < 24]

        means = df_2013_2014.groupby('month').target.mean()
        dfmain['month_mean_target'] = dfmain['month'].map(means)

        '''
        for parent item_category
        The group by to be performed on the df_sales original dataframe        

        for item_category_id
        The group by to be performed on the df_sales original dataframe        
        '''
        #parent category (disabling it for submission 30may2020_kaggle_2.ipynb) only
        #means_parent_cat = dfmain.groupby('parent_cat_id').target.mean() #calculating the mean from sales_train
        #dfmain['parent_cat_mean_target'] = dfmain['parent_cat_id'].map(means_parent_cat) #attaching the mean to the matrix dataframe (dfmain)

        #item category
        means_cat = dfmain.groupby('item_category_id').target.mean()
        dfmain['item_cat_mean_target'] = dfmain['item_category_id'].map(means_cat)

        '''
        for shop_id        
        '''
        means_shops = dfmain.groupby('shop_id').target.mean()
        dfmain['shop_id_mean_target'] = dfmain['shop_id'].map(means_shops)


        '''
        for city_id
        '''
        #mean_city = dfmain.groupby('city_id').target.mean()
        #dfmain['city_id_mean_target'] = dfmain['city_id'].map(mean_city)


        del df_2013_2014
        return dfmain
    
    def save_submission(self, model_name, df):
        '''
        The function will construct the file submission name, and save it to the specified path for submissions
        '''
        file_name =  'sub_' + model_name
        save_path = self.__submission_path + '\\' + file_name + '_' + str(self.__today) + '.csv'
        df.to_csv(save_path, index=False)
        print('Submission file saved as ' + save_path)

    def clean_duplicate_item_ids(self, df):
        t = df.groupby(['date_block_num','shop_id','item_id']).sum().reset_index()
        t = pd.merge(df,t,how='left',on=['date_block_num','shop_id','item_id'])
        del t['item_price_x'], t['item_cnt_day_x']
        t = t.rename(columns={"item_cnt_day_y": "item_cnt_day","item_price_y": "item_price"})
        t = t.drop_duplicates(['date_block_num','shop_id','item_id','item_price','item_cnt_day'],keep='first')
        return t

    def group_duplicate_categories(self, df_item_categories):
        cat_id_1 = []
        cat_name_1 = []
        cat_id_2 = []
        cat_name_2 = []
        match_perc = []

        for cat in df_item_categories['item_cat_en'].unique():
            #getting the shop_to_check
            top_match_ratio = 0
            for cat_x in df_item_categories[df_item_categories.item_cat_en != cat]['item_cat_en'].unique():
                match_ratio = fuzz.token_set_ratio(cat,cat_x)
                if match_ratio > top_match_ratio:
                    top_match_ratio=match_ratio
                    best_matched_cat = cat_x

            #get the shop_id for shop
            cat_id = int(df_item_categories[df_item_categories.item_cat_en==cat].item_category_id)
            cat_id_x = int(df_item_categories[df_item_categories.item_cat_en==best_matched_cat].item_category_id)

            cat_id_1.append(cat_id)
            cat_name_1.append(cat)
            cat_id_2.append(cat_id_x)
            cat_name_2.append(best_matched_cat)
            match_perc.append(top_match_ratio)

        cat_name_matches = np.c_[cat_id_1, cat_id_2,cat_name_1,cat_name_2, match_perc]
        cat_similarities = pd.DataFrame(data=cat_name_matches, columns=['CatA','CatB','CatAName','CatBName','Perc'])

        return cat_similarities


    def group_duplicate_shops(self, df_shops):
        shop_id_1 = []
        shop_id_2 = []
        match_perc = []
        for shop in df_shops['shop_name_en'].unique():
            #getting the shop_to_check
            top_match_ratio = 0
            for shop_x in df_shops[df_shops.shop_name_en != shop]['shop_name_en'].unique():
                match_ratio = fuzz.token_set_ratio(shop,shop_x)
                if match_ratio > top_match_ratio:
                    top_match_ratio=match_ratio
                    best_matched_shop = shop_x
                    
            #get the shop_id for shop
            shop_id = int(df_shops[df_shops.shop_name_en==shop].shop_id)
            shop_id_x = int(df_shops[df_shops.shop_name_en==best_matched_shop].shop_id)
            
            shop_id_1.append(shop_id)
            shop_id_2.append(shop_id_x)
            match_perc.append(top_match_ratio)
            

        shop_name_matches = np.c_[shop_id_1, shop_id_2, match_perc]
        shop_similarities = pd.DataFrame(data=shop_name_matches, columns=['ShopA','ShopB','Perc'])
        return shop_similarities


    def plot_sales_trend(self,ids,col,df):
        #creating filter
        df_filtered = df[df[col].isin(ids)]
        plot_title = 'Trend for ' + str(col) + ': ' + str(ids)
        
        fig, axes = plt.subplots(figsize=(19,9))
        plt.title(plot_title)    
        axes.set(xlim=(0, 33), ylim=(0,df_filtered.item_cnt_day.max()+10))
        sns.set(style='whitegrid')
        axes = sns.pointplot(x='date_block_num', y='item_cnt_day', data=df_filtered, hue=col)  

        

    def say_hello(self, strname):
        return 'hello' + strname

    
    
