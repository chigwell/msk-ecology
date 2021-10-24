#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle


# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared 
    
    def __init__(self,columns):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler()
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class for CO polution that we are going to use from here on to predict new data
class polution_CO_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model_CO','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it 
        def load_and_clean_data(self, data_file):
            
            # import the data
            user_data = pd.read_csv(data_file,delimiter=',')
            df=user_data
            
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            
            #transform data into datetime type
            df['date']=pd.to_datetime(df['date'])

            # create list of month represented by number(from 1 to 12) and add it to data
            list_months=[]
            for i in range(df.shape[0]):
                list_months.append(df['date'][i].month)
            df['season'] = list_months
            
            # create list of week days represented by number(from 1 to 7) and add it to data
            list_dayofweek=[]
            for i in range(df.shape[0]):
                list_dayofweek.append((df['date'][i].dayofweek)+1)
            df['week_day'] = list_dayofweek
            
            #removing not nessasary column date from initial data
            df=df.drop(['date'], axis=1)

            # load information about factory dencity in the city
            factory_dencity = pd.read_csv('factory_dencity.csv')
            columns_factory = ['season', 'industrial', 'electricity', 'processing', 'water_supply']
            factory_dencity = factory_dencity[columns_factory]
            
            #adding information about factory dencity in the city to our data, filtered by season column
            df=df.merge(factory_dencity, on='season')

            #load information about traffic in the city during the seasons(months)
            traffic_season_dencity=pd.read_csv('traffic_season_dencity.csv')
            columns_traffic_season = ['season', 'season_traffic']
            traffic_season_dencity = traffic_season_dencity[columns_traffic_season]
            
            #adding mentioned above info to our data, filtered by season column
            df=df.merge(traffic_season_dencity, on='season')

            #load information about traffic in the city during the day and week
            traffic_day_dencity=pd.read_csv('traffic_day_dencity.csv')
            columns_traffic_day_dencity = ['time', 'week_day', 'traffic']
            traffic_day_dencity = traffic_day_dencity[columns_traffic_day_dencity]
            
            ##adding mentioned above info to our data, filtered by 2 columns: "time","week_day"
            df=pd.merge(df,traffic_day_dencity,on=["time","week_day"],how="inner", sort=False)


            #load preproceced information about temperature inversion in the city during the day and week and seasons
            df_inversion=pd.read_csv('df_inversion.csv')
            columns_df_inversion = ['time', 'season', 'week_day', 'inversion_high200', 'inversion_high400', 'inversion_high600']
            df_inversion = df_inversion[columns_df_inversion]

            #adding mentioned above info to our data, filtered by 3 columns: "time","week_day","season" 
            df=pd.merge(df,df_inversion,on=["season","week_day", 'time'],how="inner", sort=False)
            
            #proceccing data - getting mean value for each inversion column and add it to the column
            df['inversion_high200']=df['inversion_high200'].mean()
            df['inversion_high400']=df['inversion_high400'].mean()
            df['inversion_high600']=df['inversion_high600'].mean()
            df=df.iloc[:1,:]

            #load loading the information about wind in the city in general(256meters)
            wind253=pd.read_csv('wind253.csv')
            columns_wind253 = ['time', 'season', 'week_day', '_V0_', '| V0 |']
            wind253 = wind253[columns_wind253]
            
            # #adding mentioned above info to our data, filtered by 3 columns: "time","week_day","season" 
            df=pd.merge(df,wind253,on=["season","week_day", 'time'],how="inner", sort=False)
            
            #proceccing data - getting mean value for each and add it to the column
            df['_V0_']=df['_V0_'].mean()
            df['| V0 |']=df['| V0 |'].mean()
            df=df.iloc[:1,:]

            #adding building_density information
            building_density = pd.read_csv('building_density.csv')
            columns_building_density = ['station_name', 'dencity_coef']
            building_density = building_density[columns_building_density]
            building_density.rename({'dencity_coef': 'building_dencity_coef'}, axis=1, inplace=True)
            df=df.merge(building_density, on='station_name')
            
            #proccec heo station name info(get dummies)

            GeoStation=pd.DataFrame({'station_name': ['shabalovka', 'turistskaya',
                   'spiridonovka', 'proletarski', 'marino', 'koptevskii',
                   'glebovskaya', 'butlerova', 'anohina', 'ostankino' ]})
            
            geo_column=pd.get_dummies(GeoStation['station_name'])
            col=GeoStation['station_name'].values
            geo_column=geo_column[col]

            for i in range(geo_column.shape[0]):
                if i==geo_column.shape[0]:
                    geo_column=geo_column.iloc[i-1:i,:]
                if df['station_name'][0] is geo_column.columns[i] and geo_column.iloc[i:i+1,i:i+1].iat[0,0] ==1:
                    geo_column=geo_column.iloc[i:i+1,:]
            geo_column.reset_index(drop=True, inplace=True)
            df=pd.concat([df, geo_column], sort=False, axis=1)
            df=df.drop(['station_name'], axis=1)
            df=df.drop(['ostankino'], axis=1)
            
            #reoder the columns
            columns_df=['season', 'week_day', 'time', 'industrial', 'electricity',
                   'processing', 'water_supply', 'season_traffic', 'traffic',
                   'inversion_high200', 'inversion_high400', 'inversion_high600',
                   '_V0_', '| V0 |', 'building_dencity_coef', 'shabalovka',
                   'turistskaya', 'spiridonovka', 'proletarski', 'marino',
                   'koptevskii', 'glebovskaya', 'butlerova', 'anohina', '-T-', '| V |', '_V_', 'pressure', 'humidity',
                   'precipitation' ]
            df=df[columns_df]
            df=df.iloc[:1, :]
            
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
            
            #this data for output
            raw_data = pd.read_csv('prepared_Final_data.csv')    
            self.data_mean_CO = raw_data['CO'].median()
            self.data_CO_std = np.std(raw_data['CO'])

            self.user_data = user_data
            
            
        
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # a function based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.user_data['Probability'] = self.reg.predict_proba(self.data)[:,1]   
                self.user_data ['CO'] = ((self.data_mean_CO+ self.data_CO_std) * self.reg.predict_proba(self.data)[:,1] 
                                         + (self.data_mean_CO- self.data_CO_std) * self.reg.predict_proba(self.data)[:,0:1])/2 
                
                return self.user_data
            
            
# create the special class for NO2 polution that we are going to use from here on to predict new data
class polution_NO2_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model_NO2','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            user_data = pd.read_csv(data_file,delimiter=',')
            df=user_data
            
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            df['date']=pd.to_datetime(df['date'])

            #
            list_months=[]
            for i in range(df.shape[0]):
                list_months.append(df['date'][i].month)
            df['season'] = list_months
            #
            list_dayofweek=[]
            for i in range(df.shape[0]):
                list_dayofweek.append((df['date'][i].dayofweek)+1)
            df['week_day'] = list_dayofweek

            df=df.drop(['date'], axis=1)

            #
            factory_dencity = pd.read_csv('factory_dencity.csv')
            columns_factory = ['season', 'industrial', 'electricity', 'processing', 'water_supply']
            factory_dencity = factory_dencity[columns_factory]
            #
            df=df.merge(factory_dencity, on='season')

            #
            traffic_season_dencity=pd.read_csv('traffic_season_dencity.csv')
            columns_traffic_season = ['season', 'season_traffic']
            traffic_season_dencity = traffic_season_dencity[columns_traffic_season]
            #
            df=df.merge(traffic_season_dencity, on='season')

            #
            traffic_day_dencity=pd.read_csv('traffic_day_dencity.csv')
            columns_traffic_day_dencity = ['time', 'week_day', 'traffic']
            traffic_day_dencity = traffic_day_dencity[columns_traffic_day_dencity]
            #
            df=pd.merge(df,traffic_day_dencity,on=["time","week_day"],how="inner", sort=False)


            #
            df_inversion=pd.read_csv('df_inversion.csv')
            columns_df_inversion = ['time', 'season', 'week_day', 'inversion_high200', 'inversion_high400', 'inversion_high600']
            df_inversion = df_inversion[columns_df_inversion]

            #
            df=pd.merge(df,df_inversion,on=["season","week_day", 'time'],how="inner", sort=False)
            df['inversion_high200']=df['inversion_high200'].mean()
            df['inversion_high400']=df['inversion_high400'].mean()
            df['inversion_high600']=df['inversion_high600'].mean()

            df=df.iloc[:1,:]

            #
            wind253=pd.read_csv('wind253.csv')
            columns_wind253 = ['time', 'season', 'week_day', '_V0_', '| V0 |']
            wind253 = wind253[columns_wind253]
            #
            df=pd.merge(df,wind253,on=["season","week_day", 'time'],how="inner", sort=False)
            df['_V0_']=df['_V0_'].mean()
            df['| V0 |']=df['| V0 |'].mean()

            df=df.iloc[:1,:]

            #
            building_density = pd.read_csv('building_density.csv')
            columns_building_density = ['station_name', 'dencity_coef']
            building_density = building_density[columns_building_density]

            building_density.rename({'dencity_coef': 'building_dencity_coef'}, axis=1, inplace=True)

            df=df.merge(building_density, on='station_name')

            GeoStation=pd.DataFrame({'station_name': ['shabalovka', 'turistskaya',
                   'spiridonovka', 'proletarski', 'marino', 'koptevskii',
                   'glebovskaya', 'butlerova', 'anohina', 'ostankino' ]})



            geo_column=pd.get_dummies(GeoStation['station_name'])
            col=GeoStation['station_name'].values
            geo_column=geo_column[col]

            for i in range(geo_column.shape[0]):
                if i==geo_column.shape[0]:
                    geo_column=geo_column.iloc[i-1:i,:]
                if df['station_name'][0] is geo_column.columns[i] and geo_column.iloc[i:i+1,i:i+1].iat[0,0] ==1:
                    geo_column=geo_column.iloc[i:i+1,:]
            geo_column.reset_index(drop=True, inplace=True)
            df=pd.concat([df, geo_column], sort=False, axis=1)
            df=df.drop(['station_name'], axis=1)
            df=df.drop(['ostankino'], axis=1)

            columns_df=['season', 'week_day', 'time', 'industrial', 'electricity',
                   'processing', 'water_supply', 'season_traffic', 'traffic',
                   'inversion_high200', 'inversion_high400', 'inversion_high600',
                   '_V0_', '| V0 |', 'building_dencity_coef', 'shabalovka',
                   'turistskaya', 'spiridonovka', 'proletarski', 'marino',
                   'koptevskii', 'glebovskaya', 'butlerova', 'anohina', '-T-', '| V |', '_V_', 'pressure', 'humidity',
                   'precipitation' ]
            df=df[columns_df]
            df=df.iloc[:1, :]
            
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
            
            raw_data = pd.read_csv('prepared_Final_data.csv')
            self.data_mean_NO2 = raw_data['NO2'].median()
            self.data_NO2_std = np.std(raw_data['NO2'])
            
            self.user_data = user_data
            
            
        
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.user_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.user_data ['NO2'] = ((self.data_mean_NO2+ self.data_NO2_std) * self.reg.predict_proba(self.data)[:,1] 
                                          +(self.data_mean_NO2- self.data_NO2_std)* self.reg.predict_proba(self.data)[:,0:1])/2 
               
                return self.user_data     
            
            
# create the special class for NO polution that we are going to use from here on to predict new data
class polution_NO_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model_NO','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            user_data = pd.read_csv(data_file,delimiter=',')
            df=user_data
            
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            df['date']=pd.to_datetime(df['date'])

            #
            list_months=[]
            for i in range(df.shape[0]):
                list_months.append(df['date'][i].month)
            df['season'] = list_months
            #
            list_dayofweek=[]
            for i in range(df.shape[0]):
                list_dayofweek.append((df['date'][i].dayofweek)+1)
            df['week_day'] = list_dayofweek

            df=df.drop(['date'], axis=1)

            #
            factory_dencity = pd.read_csv('factory_dencity.csv')
            columns_factory = ['season', 'industrial', 'electricity', 'processing', 'water_supply']
            factory_dencity = factory_dencity[columns_factory]
            #
            df=df.merge(factory_dencity, on='season')

            #
            traffic_season_dencity=pd.read_csv('traffic_season_dencity.csv')
            columns_traffic_season = ['season', 'season_traffic']
            traffic_season_dencity = traffic_season_dencity[columns_traffic_season]
            #
            df=df.merge(traffic_season_dencity, on='season')

            #
            traffic_day_dencity=pd.read_csv('traffic_day_dencity.csv')
            columns_traffic_day_dencity = ['time', 'week_day', 'traffic']
            traffic_day_dencity = traffic_day_dencity[columns_traffic_day_dencity]
            #
            df=pd.merge(df,traffic_day_dencity,on=["time","week_day"],how="inner", sort=False)


            #
            df_inversion=pd.read_csv('df_inversion.csv')
            columns_df_inversion = ['time', 'season', 'week_day', 'inversion_high200', 'inversion_high400', 'inversion_high600']
            df_inversion = df_inversion[columns_df_inversion]

            #
            df=pd.merge(df,df_inversion,on=["season","week_day", 'time'],how="inner", sort=False)
            df['inversion_high200']=df['inversion_high200'].mean()
            df['inversion_high400']=df['inversion_high400'].mean()
            df['inversion_high600']=df['inversion_high600'].mean()

            df=df.iloc[:1,:]

            #
            wind253=pd.read_csv('wind253.csv')
            columns_wind253 = ['time', 'season', 'week_day', '_V0_', '| V0 |']
            wind253 = wind253[columns_wind253]
            #
            df=pd.merge(df,wind253,on=["season","week_day", 'time'],how="inner", sort=False)
            df['_V0_']=df['_V0_'].mean()
            df['| V0 |']=df['| V0 |'].mean()

            df=df.iloc[:1,:]

            #
            building_density = pd.read_csv('building_density.csv')
            columns_building_density = ['station_name', 'dencity_coef']
            building_density = building_density[columns_building_density]

            building_density.rename({'dencity_coef': 'building_dencity_coef'}, axis=1, inplace=True)

            df=df.merge(building_density, on='station_name')

            GeoStation=pd.DataFrame({'station_name': ['shabalovka', 'turistskaya',
                   'spiridonovka', 'proletarski', 'marino', 'koptevskii',
                   'glebovskaya', 'butlerova', 'anohina', 'ostankino' ]})



            geo_column=pd.get_dummies(GeoStation['station_name'])
            col=GeoStation['station_name'].values
            geo_column=geo_column[col]

            for i in range(geo_column.shape[0]):
                if i==geo_column.shape[0]:
                    geo_column=geo_column.iloc[i-1:i,:]
                if df['station_name'][0] is geo_column.columns[i] and geo_column.iloc[i:i+1,i:i+1].iat[0,0] ==1:
                    geo_column=geo_column.iloc[i:i+1,:]
            geo_column.reset_index(drop=True, inplace=True)
            df=pd.concat([df, geo_column], sort=False, axis=1)
            df=df.drop(['station_name'], axis=1)
            df=df.drop(['ostankino'], axis=1)

            columns_df=['season', 'week_day', 'time', 'industrial', 'electricity',
                   'processing', 'water_supply', 'season_traffic', 'traffic',
                   'inversion_high200', 'inversion_high400', 'inversion_high600',
                   '_V0_', '| V0 |', 'building_dencity_coef', 'shabalovka',
                   'turistskaya', 'spiridonovka', 'proletarski', 'marino',
                   'koptevskii', 'glebovskaya', 'butlerova', 'anohina', '-T-', '| V |', '_V_', 'pressure', 'humidity',
                   'precipitation' ]
            df=df[columns_df]
            df=df.iloc[:1, :]
            
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
            
            raw_data = pd.read_csv('prepared_Final_data.csv')
            self.data_mean_NO = raw_data['NO'].median()
            self.data_NO_std = np.std(raw_data['NO'])
            
            self.user_data = user_data
            
            
        
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.user_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.user_data ['NO'] = ((self.data_mean_NO+ self.data_NO_std)* self.reg.predict_proba(self.data)[:,1]
                                         +(self.data_mean_NO- self.data_NO_std)* self.reg.predict_proba(self.data)[:,0:1])/2 
             
            return self.user_data     
            

#create the special class for PM10 polution that we are going to use from here on to predict new data
class polution_PM10_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model_PM10','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            user_data = pd.read_csv(data_file,delimiter=',')
            df=user_data
            
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            df['date']=pd.to_datetime(df['date'])

            #
            list_months=[]
            for i in range(df.shape[0]):
                list_months.append(df['date'][i].month)
            df['season'] = list_months
            #
            list_dayofweek=[]
            for i in range(df.shape[0]):
                list_dayofweek.append((df['date'][i].dayofweek)+1)
            df['week_day'] = list_dayofweek

            df=df.drop(['date'], axis=1)

            #
            factory_dencity = pd.read_csv('factory_dencity.csv')
            columns_factory = ['season', 'industrial', 'electricity', 'processing', 'water_supply']
            factory_dencity = factory_dencity[columns_factory]
            #
            df=df.merge(factory_dencity, on='season')

            #
            traffic_season_dencity=pd.read_csv('traffic_season_dencity.csv')
            columns_traffic_season = ['season', 'season_traffic']
            traffic_season_dencity = traffic_season_dencity[columns_traffic_season]
            #
            df=df.merge(traffic_season_dencity, on='season')

            #
            traffic_day_dencity=pd.read_csv('traffic_day_dencity.csv')
            columns_traffic_day_dencity = ['time', 'week_day', 'traffic']
            traffic_day_dencity = traffic_day_dencity[columns_traffic_day_dencity]
            #
            df=pd.merge(df,traffic_day_dencity,on=["time","week_day"],how="inner", sort=False)


            ##
            df_inversion=pd.read_csv('df_inversion.csv')
            columns_df_inversion = ['time', 'season', 'week_day', 'inversion_high200', 'inversion_high400', 'inversion_high600']
            df_inversion = df_inversion[columns_df_inversion]

            #
            df=pd.merge(df,df_inversion,on=["season","week_day", 'time'],how="inner", sort=False)
            df['inversion_high200']=df['inversion_high200'].mean()
            df['inversion_high400']=df['inversion_high400'].mean()
            df['inversion_high600']=df['inversion_high600'].mean()

            df=df.iloc[:1,:]

            #
            wind253=pd.read_csv('wind253.csv')
            columns_wind253 = ['time', 'season', 'week_day', '_V0_', '| V0 |']
            wind253 = wind253[columns_wind253]
            #
            df=pd.merge(df,wind253,on=["season","week_day", 'time'],how="inner", sort=False)
            df['_V0_']=df['_V0_'].mean()
            df['| V0 |']=df['| V0 |'].mean()

            df=df.iloc[:1,:]

            #
            building_density = pd.read_csv('building_density.csv')
            columns_building_density = ['station_name', 'dencity_coef']
            building_density = building_density[columns_building_density]

            building_density.rename({'dencity_coef': 'building_dencity_coef'}, axis=1, inplace=True)

            df=df.merge(building_density, on='station_name')

            GeoStation=pd.DataFrame({'station_name': ['shabalovka', 'turistskaya',
                   'spiridonovka', 'proletarski', 'marino', 'koptevskii',
                   'glebovskaya', 'butlerova', 'anohina', 'ostankino' ]})



            geo_column=pd.get_dummies(GeoStation['station_name'])
            col=GeoStation['station_name'].values
            geo_column=geo_column[col]

            for i in range(geo_column.shape[0]):
                if i==geo_column.shape[0]:
                    geo_column=geo_column.iloc[i-1:i,:]
                if df['station_name'][0] is geo_column.columns[i] and geo_column.iloc[i:i+1,i:i+1].iat[0,0] ==1:
                    geo_column=geo_column.iloc[i:i+1,:]
            geo_column.reset_index(drop=True, inplace=True)
            df=pd.concat([df, geo_column], sort=False, axis=1)
            df=df.drop(['station_name'], axis=1)
            df=df.drop(['ostankino'], axis=1)

            columns_df=['season', 'week_day', 'time', 'industrial', 'electricity',
                   'processing', 'water_supply', 'season_traffic', 'traffic',
                   'inversion_high200', 'inversion_high400', 'inversion_high600',
                   '_V0_', '| V0 |', 'building_dencity_coef', 'shabalovka',
                   'turistskaya', 'spiridonovka', 'proletarski', 'marino',
                   'koptevskii', 'glebovskaya', 'butlerova', 'anohina', '-T-', '| V |', '_V_', 'pressure', 'humidity',
                   'precipitation' ]
            df=df[columns_df]
            df=df.iloc[:1, :]
            
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
            
            raw_data = pd.read_csv('prepared_Final_data.csv')
            self.data_mean_PM10 = raw_data['PM10'].median()
            self.data_PM10_std = np.std(raw_data['PM10'])
            
            self.user_data = user_data
            
            
        
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred

        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.user_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.user_data ['PM10'] = ((self.data_mean_PM10+ self.data_PM10_std) * self.reg.predict_proba(self.data)[:,1]
                                           + (self.data_mean_PM10- self.data_PM10_std) * self.reg.predict_proba(self.data)[:,0:1])/2 
               
                return self.user_data   
            

#reate the special class for PM2.5 polution that we are going to use from here on to predict new data
class polution_PM25_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model_PM25','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            user_data = pd.read_csv(data_file,delimiter=',')
            df=user_data
            
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            df['date']=pd.to_datetime(df['date'])

            #
            list_months=[]
            for i in range(df.shape[0]):
                list_months.append(df['date'][i].month)
            df['season'] = list_months
            #
            list_dayofweek=[]
            for i in range(df.shape[0]):
                list_dayofweek.append((df['date'][i].dayofweek)+1)
            df['week_day'] = list_dayofweek

            df=df.drop(['date'], axis=1)

            #
            factory_dencity = pd.read_csv('factory_dencity.csv')
            columns_factory = ['season', 'industrial', 'electricity', 'processing', 'water_supply']
            factory_dencity = factory_dencity[columns_factory]
            #
            df=df.merge(factory_dencity, on='season')

            #
            traffic_season_dencity=pd.read_csv('traffic_season_dencity.csv')
            columns_traffic_season = ['season', 'season_traffic']
            traffic_season_dencity = traffic_season_dencity[columns_traffic_season]
            #
            df=df.merge(traffic_season_dencity, on='season')

            #
            traffic_day_dencity=pd.read_csv('traffic_day_dencity.csv')
            columns_traffic_day_dencity = ['time', 'week_day', 'traffic']
            traffic_day_dencity = traffic_day_dencity[columns_traffic_day_dencity]
            #
            df=pd.merge(df,traffic_day_dencity,on=["time","week_day"],how="inner", sort=False)


            #
            df_inversion=pd.read_csv('df_inversion.csv')
            columns_df_inversion = ['time', 'season', 'week_day', 'inversion_high200', 'inversion_high400', 'inversion_high600']
            df_inversion = df_inversion[columns_df_inversion]

            #
            df=pd.merge(df,df_inversion,on=["season","week_day", 'time'],how="inner", sort=False)
            df['inversion_high200']=df['inversion_high200'].mean()
            df['inversion_high400']=df['inversion_high400'].mean()
            df['inversion_high600']=df['inversion_high600'].mean()

            df=df.iloc[:1,:]

            #
            wind253=pd.read_csv('wind253.csv')
            columns_wind253 = ['time', 'season', 'week_day', '_V0_', '| V0 |']
            wind253 = wind253[columns_wind253]
            #
            df=pd.merge(df,wind253,on=["season","week_day", 'time'],how="inner", sort=False)
            df['_V0_']=df['_V0_'].mean()
            df['| V0 |']=df['| V0 |'].mean()

            df=df.iloc[:1,:]

            #
            building_density = pd.read_csv('building_density.csv')
            columns_building_density = ['station_name', 'dencity_coef']
            building_density = building_density[columns_building_density]

            building_density.rename({'dencity_coef': 'building_dencity_coef'}, axis=1, inplace=True)

            df=df.merge(building_density, on='station_name')

            GeoStation=pd.DataFrame({'station_name': ['shabalovka', 'turistskaya',
                   'spiridonovka', 'proletarski', 'marino', 'koptevskii',
                   'glebovskaya', 'butlerova', 'anohina', 'ostankino' ]})



            geo_column=pd.get_dummies(GeoStation['station_name'])
            col=GeoStation['station_name'].values
            geo_column=geo_column[col]

            for i in range(geo_column.shape[0]):
                if i==geo_column.shape[0]:
                    geo_column=geo_column.iloc[i-1:i,:]
                if df['station_name'][0] is geo_column.columns[i] and geo_column.iloc[i:i+1,i:i+1].iat[0,0] ==1:
                    geo_column=geo_column.iloc[i:i+1,:]
            geo_column.reset_index(drop=True, inplace=True)
            df=pd.concat([df, geo_column], sort=False, axis=1)
            df=df.drop(['station_name'], axis=1)
            df=df.drop(['ostankino'], axis=1)

            columns_df=['season', 'week_day', 'time', 'industrial', 'electricity',
                   'processing', 'water_supply', 'season_traffic', 'traffic',
                   'inversion_high200', 'inversion_high400', 'inversion_high600',
                   '_V0_', '| V0 |', 'building_dencity_coef', 'shabalovka',
                   'turistskaya', 'spiridonovka', 'proletarski', 'marino',
                   'koptevskii', 'glebovskaya', 'butlerova', 'anohina', '-T-', '| V |', '_V_', 'pressure', 'humidity',
                   'precipitation' ]
            df=df[columns_df]
            df=df.iloc[:1, :]
            
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
            
            raw_data = pd.read_csv('prepared_Final_data.csv')    
            self.data_mean_PM25 = raw_data['PM2.5'].median()
            self.data_PM25_std = np.std(raw_data['PM2.5'])
            
            self.user_data = user_data
            
            
        
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
         
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.user_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.user_data ['PM2.5'] = ((self.data_mean_PM25+ self.data_PM25_std) * self.reg.predict_proba(self.data)[:,1]
                                            +(self.data_mean_PM25- self.data_PM25_std)* self.reg.predict_proba(self.data)[:,0:1])/2 

                return self.user_data   

