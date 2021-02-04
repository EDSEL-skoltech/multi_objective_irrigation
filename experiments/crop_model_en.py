 #!/usr/bin/env python3
import os, sys
import argparse
import numpy as np
import json
import multiprocessing
import time
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta

import pcse
from pcse.models import Wofost71_WLP_FD
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider
from pcse.engine import Engine
import datetime as dt


class Irrigation():
    """
    Update: Class for all crop simulations with WOFOST model 


    Class for irrigation optimization

    input: np.array of days and ammount of water
    
    Example: [12,45,43] - days
             [13,23,23] - water in sm
             np.np.concatenate(days, water)
    output: float (yield)
    
    """
    def __init__(self):
        
    
        self.date_crop_start = None
        self.date_crop_end = None
        self.range_irrigation_season = []
        self.optimal_dates_irrigation = None
        self.optimal_ammounts = None
    
        #path to soil dats
        self.data_dir = os.path.join(os.getcwd(), "data")
        
        self.weather = None
        self.user_parameters = None
        self.total_ammount_of_losed_water = None

        self.effective_irrigation = None

        self.path_to_CSV_database = '../NASA_TEST/'
        self._csv_weather_database_boarders = {'latitude_min':10,
                                              'latitude_max':90,
                                              'longitude_min':10,
                                              'longitude_max':160}
    
    def irrigation(self, x):
        """
        np.array with dates and amounts of water
        
        Ex.: [31, 42, 54, 10, 15, 16]
        
        """
        
        assert len(x)%2==0, 'not symmetric'
        half=len(x)//2
        dates = x[:half]
        amounts = x[half:]
        local_range = pd.date_range(start=self.date_crop_start,end=self.date_crop_end)


        dates_of_irrigation =[str(local_range[date])[:10] for date in dates]
        ammounts_of_irrigation=amounts
        return dates_of_irrigation, ammounts_of_irrigation
    

    def load_model(self):
        """
        Function to load input soil, site and crop parameters data from yaml files
        
        """
        
        crop = YAMLCropDataProvider()
#         soil = CABOFileReader(os.path.join(self.data_dir, "ec3.soil"))
        
        soil = CABOFileReader(os.path.join(self.data_dir, "wofost_npk.soil"))
        site = CABOFileReader(os.path.join(self.data_dir, "wofost_npk.site"))
        site['CO2']=360.0
#         site = WOFOST71SiteDataProvider(WAV=100,CO2=360)

        #parameters for model
        self.parameterprovider = ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)


    def run_simulation(self, crop_calendar):
        
        # import yaml
        # agromanagement = yaml.load(crop_calendar)
        agromanagement = yaml.safe_load(crop_calendar)

        wofost = Wofost71_WLP_FD(self.parameterprovider, self.weather, agromanagement)
        wofost.run_till_terminate()

        self.output = wofost.get_output()
        return self.output[-1]['TWSO'] 
     
    def run_simulation_manager(self, agromanagement):
        # from pcse.engine import Engine
        wofost = Wofost71_WLP_FD(self.parameterprovider, self.weather, agromanagement)
        # wofost = Engine(self.parameterprovider,  self.weather, agromanagement, config=os.path.join(self.data_dir, "Wofost71_NPK_grol.conf"))
        wofost.run_till_terminate()
        water_losed_into_deep_horizont = wofost.get_terminal_output()
        sum_water = water_losed_into_deep_horizont['PERCT'] + water_losed_into_deep_horizont['LOSST']

        # for paper - estimate effictive irrigation 

        self.effective_irrigation = water_losed_into_deep_horizont['RAINT']



        self.total_ammount_of_losed_water = sum_water
        self.output = wofost.get_output()
    
        return self.output[-1]['TWSO'] 
    
    
    def weather_loader(self,path_CSV_dir, latitude, longitude):
        """
        Main fun to load weather 
        
        If we have CSV file - load CSV, 
        
        else: Load from NASA
        
        """

        self.path_to_CSV_database = path_CSV_dir


        # latitude_min = self._csv_weather_database_boarders['latitude_min']
        # latitude_max = self._csv_weather_database_boarders['latitude_max']
        # longitude_min = self._csv_weather_database_boarders['longitude_min']
        # longitude_max = self._csv_weather_database_boarders['longitude_max']
        # if latitude_min <= latitude < latitude_max and longitude_min <= longitude < longitude_max:
            # If in range of our database - load CSV file from database
        path = self.path_to_CSV_database + f'/NASA_weather_latitude_{latitude}_longitude_{longitude}.csv'
        # print('path and file', path)
        if os.path.exists(path):

            print('LOAD FROM LOCAL CSV WEATHER DATABASE')
            # path = self.path_to_CSV_database + f'NASA_weather_latitude_{latitude}_longitude_{longitude}_TEST.csv'
            weather = pcse.fileinput.CSVWeatherDataProvider(path)
            self.weather = weather
            self.NASA_start_year = weather.first_date.year+1
            
            ### WEATHER YEAR TEST
            self.NASA_last_year = weather.last_date.year-1

            #self.NASA_last_year = weather.last_date.year
        else:
            print('No such directory or CSV file')
            print('DOWNLOAD FROM NASA POWER database (1-5 mins)')

            # Test load from NASA POWER and save as  csv and after load
            path = self.path_to_CSV_database
            try: 
                start_time = time.time()
                #API request to NASA database
                weather = NASAPowerWeatherDataProvider(latitude, longitude, force_update=True)

                # Print done if downloaded
                print('____DONE_____','latitude',latitude, 'longitude',longitude,'____')

                # export pcse.weather format to pandas df
                df_weather = pd.DataFrame(weather.export())


                #print('initial number of days:', len(df_weather))

                #create full range of dates
                r = pd.date_range(start=df_weather.DAY.min(), end=df_weather.DAY.max())


                #extend range of dates
                full_range_weather = df_weather.set_index('DAY').reindex(r).rename_axis('DAY').reset_index()
                missing_days = (full_range_weather.isna()).sum().sum()

                print('num_of_missing_days', missing_days)

                #fill weather with fill forward method in pandas
                filled_weather = full_range_weather.fillna(method='ffill', axis=0)
                
                #save as csv file

                filled_weather=filled_weather[['DAY', 'IRRAD', 'TMIN', 'TMAX', 'VAP', 'WIND', 'RAIN']]
                filled_weather['SNOWDEPTH'] = 'NaN'
                filled_weather[['IRRAD']] = filled_weather[['IRRAD']]/1000.
                filled_weather[['VAP']] = filled_weather[['VAP']]/10.
                filled_weather.DAY=filled_weather.DAY.dt.strftime('%Y%m%d')


                text = open(path+"pattern.csv", "r")
                text = ''.join([i for i in text]).replace("1111", str(weather.longitude))
                text = ''.join([i for i in text]).replace("2222", str(weather.latitude))
                text = ''.join([i for i in text]).replace("3333", str(weather.elevation))
                text = ''.join([i for i in text]).replace("4444", str(weather.angstA))
                text = ''.join([i for i in text]).replace("5555", str(weather.angstB))
                x = open(path+f'NASA_weather_latitude_{latitude}_longitude_{longitude}.csv',"w")
                x.writelines(text)
                x.close()

                path_to_save_csv_file = path+f'NASA_weather_latitude_{latitude}_longitude_{longitude}.csv'
                filled_weather.to_csv(path_to_save_csv_file, mode='a', header=False, index=False)

                #add info to weather database and save it to csv

                print('time in sec', time.time() - start_time)

                #LOAD WEATHER as csv file
                weather = pcse.fileinput.CSVWeatherDataProvider(path_to_save_csv_file)
                self.weather = weather
                self.NASA_start_year = weather.first_date.year+1
                

                self.NASA_last_year = weather.last_date.year-1
            except Exception:
                info = traceback.format_exc()
                print(info)
            # self.NASA_weather_data_loader(latitude, longitude)


    
    def update_csv_NASA_weather_database(self, path_CSV_dir, latitude_min, latitude_max, longitude_min, longitude_max):
        """
        function for downloading NASA weather and creating csv files in folders for future simulation 

        Input: path, latitude_min, latitude_max, longitude_min, longitude_max
        Output: CSV files in dir 

        """
        #import time
        longitude_array = np.arange(longitude_min,longitude_max,step=1)
        latitude_array = np.arange(latitude_min,latitude_max,step=1)


        col_names=['latitude', 'longitude', 'Num_of_missing_dates', 'start_date', 'last_date']
        weather_database = pd.DataFrame(data = np.zeros([latitude_array.shape[0]*longitude_array.shape[0], len(col_names)]),  columns=col_names)
        weather_database.start_date = pd.to_datetime(weather_database.start_date)
        weather_database.last_date = pd.to_datetime(weather_database.last_date)

        #path = '/gdrive/My Drive/NASA_CSV'
        path = path_CSV_dir


        i = 0

        for latitude in latitude_array:
            for longitude in longitude_array:
                start_time = time.time()
                #API request to NASA database
                weather = NASAPowerWeatherDataProvider(latitude, longitude, force_update=True)

                # Print done if downloaded
                print('____DONE_____','latitude',latitude, 'longitude',longitude,'____')

                # export pcse.weather format to pandas df
                df_weather = pd.DataFrame(weather.export())


                #print('initial number of days:', len(df_weather))

                #create full range of dates
                r = pd.date_range(start=df_weather.DAY.min(), end=df_weather.DAY.max())


                #extend range of dates
                full_range_weather = df_weather.set_index('DAY').reindex(r).rename_axis('DAY').reset_index()
                missing_days = (full_range_weather.isna()).sum().sum()

                print('num_of_missing_days', missing_days)

                #fill weather with fill forward method in pandas
                filled_weather = full_range_weather.fillna(method='ffill', axis=0)
                ##save as csv file
                #filled_weather.to_csv(path+f'/NASA_weather_latitude_{latitude}_longitude{longitude}.csv', index=False)

                # filled_weather = pd.read_csv('../NASA_test/NASA_weather_latitude_30_longitude40.csv')
                # filled_weather.DAY = pd.to_datetime(loaded_weather.DAY)
                filled_weather=filled_weather[['DAY', 'IRRAD', 'TMIN', 'TMAX', 'VAP', 'WIND', 'RAIN']]
                filled_weather['SNOWDEPTH'] = 'NaN'
                filled_weather[['IRRAD']] = filled_weather[['IRRAD']]/1000.
                filled_weather[['VAP']] = filled_weather[['VAP']]/10.
                filled_weather.DAY=filled_weather.DAY.dt.strftime('%Y%m%d')


                text = open(path+"pattern.csv", "r")
                text = ''.join([i for i in text]).replace("1111", str(weather.longitude))
                text = ''.join([i for i in text]).replace("2222", str(weather.latitude))
                text = ''.join([i for i in text]).replace("3333", str(weather.elevation))
                text = ''.join([i for i in text]).replace("4444", str(weather.angstA))
                text = ''.join([i for i in text]).replace("5555", str(weather.angstB))
                x = open(path+f'NASA_weather_latitude_{latitude}_longitude_{longitude}_TEST.csv',"w")
                x.writelines(text)
                x.close()


                filled_weather.to_csv(path+f'NASA_weather_latitude_{latitude}_longitude_{longitude}_TEST.csv', mode='a', header=False, index=False)


                #add info to weather database and save it to csv
                list_to_add = [latitude, longitude, missing_days, weather.first_date, weather.last_date ]
                weather_database.iloc[i,:] = list_to_add
                i += 1
                weather_database.to_csv(path+'weather_database.csv', mode='a')

                print('time in sec', time.time() - start_time)

        
    def NASA_weather_data_loader(self, latitude,longitude):
        
        """
        Download weather from NASA database and fill missing values
        
        Input: latitude,longitude (int)
        
        Output: Weather database for 30-40 last years
        
        """

        weather = NASAPowerWeatherDataProvider(latitude, longitude, force_update=True)
        df_weather = pd.DataFrame(weather.export())

        r = pd.date_range(start=df_weather.DAY.min(), end=df_weather.DAY.max())
        full_range_weather = df_weather.set_index('DAY').reindex(r).rename_axis('DAY').reset_index()
        filled_weather = full_range_weather.fillna(method='ffill', axis=0)
        weather._make_WeatherDataContainers(filled_weather.to_dict(orient="records"))
    
        self.NASA_start_year = weather.first_date.year+1
        self.NASA_last_year = weather.last_date.year-1
        self.weather = weather


    def agromanager_writer(self, crop_name, dates_irrigation, dates_npk, amounts, npk_list):
        """ 
        Fun to add new irrigation events in agrocalendar
        
        Input: dates - list of date in str format (ex. 2006-07-10)
            amounts - list of water mm in str format (ex. '10')
            
        Example: #add example
        
        """
        import datetime as dt
        self.date_start = (dt.datetime.strptime(self.date_crop_start, '%Y-%m-%d') - dt.timedelta(days=2)).strftime(format='%Y-%m-%d')



        dict_of_crop_sorts = {'barley':'Spring_barley_301',
                    'cassava':'Cassava_VanHeemst_1988',
                    'chickpea':'Chickpea_VanHeemst_1988',
                    'cotton':'Cotton_VanHeemst_1988',
                    'cowpea':'Cowpea_VanHeemst_1988',
                    'fababean':'Faba_bean_801',
                    'groundnut':'Groundnut_VanHeemst_1988',
                    'maize':'Maize_VanHeemst_1988',
                    'millet':'Millet_VanHeemst_1988',
                    'mungbean':'Mungbean_VanHeemst_1988',
                    'pigeonpea':'Pigeonpea_VanHeemst_1988',
                    'potato':'Potato_701',
                    'rapeseed':'Oilseed_rape_1001',
                    'rice':'Rice_501',
                    'sorghum':'Sorghum_VanHeemst_1988',
                    'soybean':'Soybean_901',
                    'sugarbeet':'Sugarbeet_601',
                    'sugarcane':'Sugarcane_VanHeemst_1988',
                    'sunflower':'Sunflower_1101',
                    'sweetpotato':'Sweetpotato_VanHeemst_1988',
                    'tobacco':'Tobacco_VanHeemst_1988',
                    'wheat':'Winter_wheat_101'}


        english_crop_name = crop_name


        english_sort_name = dict_of_crop_sorts[english_crop_name]

        #Generate crop parameters dict for future process
        crop_data = {
        'start_moment':self.date_start,
        'crop_name': english_crop_name,
        'crop_full_name':english_sort_name,
        'crop_start_date': self.date_crop_start,
        'crop_end_date': self.date_crop_end,
        'events_irrigation': [],
        'events_npk':[]
        }
        
        #check two or more irrigation events for one day - it's problem for model 

        if len(set(dates_irrigation)) != len(dates_irrigation):
            dates_irrigation=list(set(dates_irrigation))
            amounts = amounts[:len(set(dates_irrigation))]


        if len(set(dates_npk)) != len(dates_npk):
            dates_npk=list(set(dates_npk))
            amounts = amounts[:len(set(dates_npk))]
        
        crop_data['events_irrigation'] = [{date: amount for (date, amount) in zip(dates_irrigation, amounts)}]
        
        crop_data['events_npk']=[{date: npk for (date, npk) in zip(dates_npk, npk_list)}]
        
        
        template = """        
- 2000-01-01:
    CropCalendar:
        crop_name: sugarbeet
        variety_name: Sugarbeet_601
        crop_start_date: 2000-02-02
        crop_start_type: emergence
        crop_end_date: 2000-03-03
        crop_end_type: harvest
        max_duration: 300
    TimedEvents:
    -   event_signal: irrigate
        name: Irrigation application table
        comment: All irrigation amounts in cm
        events_table: 
        - 2018-07-07: {amount: 10, efficiency: 0.7}
    -   event_signal: apply_npk
        name:  Timed N/P/K application table
        comment: All fertilizer amounts in kg/ha
        events_table:
        - 2000-01-10: {N_amount : 10, P_amount: 5, K_amount: 2}
    StateEvents: null""" 
                
        crop_start = yaml.safe_load(crop_data['crop_start_date'])
        crop_end = yaml.safe_load(crop_data['crop_end_date'])

        agromanag = yaml.safe_load(template)
        agromanag[0][crop_start - dt.timedelta(days=2)] = agromanag[0].pop(dt.date(2000, 1, 1)) 
        x = (list(agromanag[0].items())[0][0])
        agromanag[0][x]['CropCalendar']['crop_name'] = english_crop_name
        agromanag[0][x]['CropCalendar']['variety_name'] = english_sort_name
        agromanag[0][x]['CropCalendar']['crop_start_date'] = crop_start
        agromanag[0][x]['CropCalendar']['crop_end_date'] = crop_end
        agromanag[0][x]['TimedEvents'][0]['events_table'].clear()
        agromanag[0][x]['TimedEvents'][1]['events_table'].clear()

        if bool(crop_data['events_irrigation'][0]):
            for date,amount in zip(dates_irrigation, amounts):
                agromanag[0][x]['TimedEvents'][0]['events_table'].append({yaml.safe_load(date):{'amount': float(amount), 'efficiency': 0.7}})
            if bool(crop_data['events_npk'][0]):
                for date, npk in zip(dates_npk, npk_list):
                    agromanag[0][x]['TimedEvents'][1]['events_table'].append({yaml.safe_load(date):{'N_amount' : npk[0], 'P_amount': npk[1], 'K_amount': npk[2], 'N_recovery':0.7, 'P_recovery':0.7, 'K_recovery':0.7}})
            else:
                #agromanagement[0][x]['TimedEvents'][:1]
                agromanag[0][x]['TimedEvents'].pop()
        else:
            agromanag[0][x]['TimedEvents'] = None
        return agromanag
    
    def crop_hpc(self, year):
        self.date_crop_start = self.year_changer(self.user_parameters['crop_start'],year)
        self.date_crop_end = self.year_changer(self.user_parameters['crop_end'],year)
        
        dates_irrigation, amounts = self.user_parameters['irrigation_events'], self.user_parameters['irrigation_ammounts']
        dates_irrigation = [self.year_changer(obj, year) for obj in dates_irrigation]
        
        dates_npk, npk_list = self.user_parameters['npk_events'], self.user_parameters['npk']
        dates_npk = [self.year_changer(obj, year) for obj in dates_npk]
        agromanagement = self.agromanager_writer(self.user_parameters['crop_name'], dates_irrigation, dates_npk, amounts, npk_list)

        self.load_model()
        self.run_simulation_manager(agromanagement)
        output = pd.DataFrame(self.output).set_index("day")
        return output
    
    

    def irrigation_dates(self, x):
        """
        np.array with dates
        
        Ex.: [31, 42, 54, 10, 15, 16]
        
        """
        dates = x
        local_range = pd.date_range(start=self.user_parameters['crop_start'],end=self.user_parameters['crop_end'])
        dates_of_irrigation =[str(local_range[date])[:10] for date in dates]
        return dates_of_irrigation

    def year_changer(self, obj, year):
        
        """
        Util function to change user year to new year
        
        
        Input: str - date event in format '%Y-%m-%d'
        
        Output: str - new date event in format '%Y-%m-%d'
        
        """
        type_of_dt = '%Y-%m-%d'
        year_delta = int(self.user_parameters['crop_end'][:4]) - year

        dt_date_crop_start=dt.datetime.strptime(obj, type_of_dt) - relativedelta(years=year_delta)
        updated_date = dt.datetime.strftime(dt_date_crop_start, type_of_dt)
    
        return updated_date

    
class Optimization(Irrigation):
    def __init__(self):
        super().__init__()
        self.year=None
        self.optimal_dates_irrigation = None
        self.num_process = 12
        self.optimizer_counter = 0
        self.container_of_mean_yields = []
        self.container_of_mean_water_loss = []
        self.container_of_irrigation_amount = []
        self.irrigation_dates_for_many_years_optim = None
        

    def multiobjective(self, x):
        """
        Minimize multiobjective function to define 
        best dates and ammounts of water for 20 years
        """
        x_dates = x[:len(self.user_parameters['irrigation_events'])]
        x_ammounts = x[len(self.user_parameters['irrigation_events']):]
        amounts = [float(i) for i in x_ammounts]


        inputs_years = np.arange(self.NASA_start_year, self.NASA_start_year+20)

        self.irrigation_dates_for_many_years_optim = self.irrigation_dates(x_dates)
        crop_sim_for_20_years = []
        water_loss_for_20_years = []
        for year in inputs_years:
            #change year from json-year to historical
            self.date_crop_start = self.year_changer(self.user_parameters['crop_start'],year)
            self.date_crop_end = self.year_changer(self.user_parameters['crop_end'],year)
            #convet dates from int to dt.datetime
            dates_irrigation = self.irrigation_dates(x_dates)
            #Setup irrigation ammount
    
            dates_irrigation = [self.year_changer(obj, year) for obj in dates_irrigation]
            dates_npk, npk_list = self.user_parameters['npk_events'], self.user_parameters['npk']
            dates_npk = [self.year_changer(obj, year) for obj in dates_npk]
            agromanagement = self.agromanager_writer(self.user_parameters['crop_name'], dates_irrigation, dates_npk, amounts, npk_list)
            self.load_model()
            self.run_simulation_manager(agromanagement)
            output = pd.DataFrame(self.output).set_index("day")
            
            ## append to list loss
            crop_sim_for_20_years.append(output)
            water_loss_for_20_years.append(self.total_ammount_of_losed_water)
            
        #select only last day crop yield 
        yield_of_crop_sim_for_20_years = [(crop_sim_for_20_years[i]['TWSO'][-1]/1000) for i in range(len(crop_sim_for_20_years))]
        # calculate mean
        out_yield = np.mean(yield_of_crop_sim_for_20_years)
        out_water_loss = np.mean(water_loss_for_20_years)
        irrigation_sum = np.sum(amounts)
        self.container_of_irrigation_amount.append(irrigation_sum)
        self.container_of_mean_water_loss.append(out_water_loss)
        self.container_of_mean_yields.append(out_yield)
        
        #counter for optimizer
        self.optimizer_counter += 1
        return -out_yield, out_water_loss ## check this in out_yield -- MAX, out_water -- MIN !!!!
    
    



if __name__ == '__main__':
    #main_crop_model_memory.py --path_to_data_dir ../pcse_notebooks/data/soil --path_to_user_file input_agro_calendar.json --path_to_CSV_weather /Users/mikhailgasanov/Documents/machine_learning/NASA_CSV --plot_name first_plot.png
    #agrpareser to setup dir of data and user JSON file 
    parser = argparse.ArgumentParser(description='Parser_of_input_data')
    parser.add_argument('--path_to_data_dir', type=str, default="./",help='Path to data with soil, crop and other parameter files', required=True)
    parser.add_argument('--path_to_user_file', type=str, default="input_agro_calendar.json",help='JSON file with user input parameters', required=True)
    parser.add_argument('--path_to_CSV_weather', type=str, default="./",help='Path to dir with CSV weather database', required=True)
    parser.add_argument('--plot_name', type=str, default='plot.png', help='Resulted plot name')
    parser.add_argument('--num_cpu', type=int, default=2, help='Available num of CPUs on your cluster')
    args = parser.parse_args()

    WOFOST = Irrigation()
    print('Done!')
