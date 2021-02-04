from crop_model_en import Irrigation

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
        self.contatiner_for_dates = []
        

    def minimize_function_20_years(self, x):
        """
        Minimize this to define optimal day for irrigation for 20 first years
        """
        inputs_years = np.arange(self.NASA_start_year, self.NASA_start_year+20)
        #dates_irrigation = self.irrigation_dates(x)
        self.irrigation_dates_for_many_years_optim = self.irrigation_dates(x)
        crop_sim_for_20_years = []
        for year in inputs_years:
            #change year from json-year to historical
            self.date_crop_start = self.year_changer(self.user_parameters['crop_start'],year)
            self.date_crop_end = self.year_changer(self.user_parameters['crop_end'],year)
            #convet dates from int to dt.datetime
            dates_irrigation = self.irrigation_dates(x)
            #Setup irrigation ammount
            amounts = [3. for _ in range(len(dates_irrigation))]
            dates_irrigation = [self.year_changer(obj, year) for obj in dates_irrigation]
            dates_npk, npk_list = self.user_parameters['npk_events'], self.user_parameters['npk']
            dates_npk = [self.year_changer(obj, year) for obj in dates_npk]
            agromanagement = self.agromanager_writer(self.user_parameters['crop_name'], dates_irrigation, dates_npk, amounts, npk_list)
            self.load_model()
            self.run_simulation_manager(agromanagement)
            output = pd.DataFrame(self.output).set_index("day")
            crop_sim_for_20_years.append(output)
        #select only last day crop yield 
        yield_of_crop_sim_for_20_years = [crop_sim_for_20_years[i]['TWSO'][-1] for i in range(len(crop_sim_for_20_years))]
        # calculate mean
        out = np.mean(yield_of_crop_sim_for_20_years)
        return -out    
    def crop_hpc_20_years(self, year):
        self.date_crop_start = self.year_changer(self.user_parameters['crop_start'],year)
        self.date_crop_end = self.year_changer(self.user_parameters['crop_end'],year)
        ## main dif here, we use self.irrigation dates instead of user_parameters
        dates_irrigation = self.irrigation_dates_for_many_years_optim
        amounts = [3. for _ in range(len(dates_irrigation))]
        dates_irrigation = [self.year_changer(obj, year) for obj in dates_irrigation]
        
        dates_npk, npk_list = self.user_parameters['npk_events'], self.user_parameters['npk']
        dates_npk = [self.year_changer(obj, year) for obj in dates_npk]
        agromanagement = self.agromanager_writer(self.user_parameters['crop_name'], dates_irrigation, dates_npk, amounts, npk_list)

        self.load_model()
        self.run_simulation_manager(agromanagement)
        output = pd.DataFrame(self.output).set_index("day")
        return output

    
        
        
    def optimized_crop_hpc(self, year):
        
        self.date_crop_start = self.year_changer(self.user_parameters['crop_start'],year)
        self.date_crop_end = self.year_changer(self.user_parameters['crop_end'],year)

        amounts = self.user_parameters['irrigation_ammounts']
        
        dates_irrigation = [self.year_changer(obj, year) for obj in self.optimal_dates_irrigation]
        dates_npk, npk_list = self.user_parameters['npk_events'], self.user_parameters['npk']
        dates_npk = [self.year_changer(obj, year) for obj in dates_npk]
        agromanagement = self.agromanager_writer(self.user_parameters['crop_name'], dates_irrigation, dates_npk, amounts, npk_list)

        self.load_model()
        self.run_simulation_manager(agromanagement)
        output = pd.DataFrame(self.output).set_index("day")
        return output

    def multiobjective(self, x):
        """
        Minimize multiobjective function to define 
        best dates and amounts of water for 20 years
        """

        x_dates = x[:len(self.user_parameters['irrigation_events'])]
        x_ammounts = x[len(self.user_parameters['irrigation_events']):]
        amounts = [float(i) for i in x_ammounts]
        inputs_years = np.arange(self.NASA_start_year, self.NASA_start_year+20)
        #dates_irrigation = self.irrigation_dates(x_dates)
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
    
    
        
    
    
if __name__ == "__main__":
        
    import pandas as pd
    import numpy as np
    import datetime as dt
    import json
    import argparse
    import os
    import matplotlib.pyplot as plt




    parser = argparse.ArgumentParser(description='Parser_of_input_data')
    parser.add_argument('--path_to_data_dir', type=str, default="/Users/mikhailgasanov/Documents/GIT/agro_rl/pcse_notebooks/data/soil",help='Path to data with soil, crop and other parameter files', required=True)
    parser.add_argument('--path_to_user_file', type=str, default="input_agro_calendar.json",help='JSON file with user input parameters', required=True)
    parser.add_argument('--path_to_CSV_weather', type=str, default="/Users/mikhailgasanov/Documents/machine_learning/NASA_CSV",help='Path to dir with CSV weather database', required=True)
    parser.add_argument('--plot_name', type=str, default='first_plot.png', help='Resulted plot name')
    parser.add_argument('--num_generation', type=int, default=20, help='Num of generation in genetic evolution')
    parser.add_argument('--path_to_npy_files', type=str, default='./npy_files/', help='path to save files with data')
    args = parser.parse_args()


    WOFOST = Optimization()
    path_to_user_file = args.path_to_user_file
    with open(path_to_user_file, 'r') as f:
        WOFOST.user_parameters = json.load(f)


    def round_geoposition(x, prec=1, base=.5):
        return round(base * round(float(x)/base),prec)
    latitude = round_geoposition(WOFOST.user_parameters['latitude'])
    longitude = round_geoposition(WOFOST.user_parameters['longitude'])
    
    
    crop_name = WOFOST.user_parameters['crop_name']

    #load historical weather data
    path_CSV_dir = args.path_to_CSV_weather
    WOFOST.weather_loader(path_CSV_dir, latitude, longitude)
    WOFOST.data_dir = args.path_to_data_dir
    # WOFOST.data_dir = '../util/input_data/'
    crop_results=[]




    max_number_of_days = len(pd.date_range(start=WOFOST.user_parameters['crop_start'],end=WOFOST.user_parameters['crop_end']))
    x_lower = ([1]*len(WOFOST.user_parameters['irrigation_events'])+[1]*len(WOFOST.user_parameters['irrigation_ammounts']))
    x_upper = ([max_number_of_days-5]*len(WOFOST.user_parameters['irrigation_events'])+[15]*len(WOFOST.user_parameters['irrigation_ammounts']))
    num_of_var = len(x_lower)
    num_generation = args.num_generation
    for i in range(num_generation):
        print('epoch', i)
        random_dates = np.random.choice(np.arange(1, 144), size=len(WOFOST.user_parameters['irrigation_events']), replace=False)
        random_water = np.random.choice(np.arange(1, 15), size=len(WOFOST.user_parameters['irrigation_events']), replace=False)
        random_x = np.concatenate([random_dates, random_water])
        _, _ = WOFOST.multiobjective(random_x)
    print('Start search for optimal solution!')

    

    # Save data for plots 





    # Save data for plots 
    path_to_folder = args.path_to_npy_files
    if os.path.isdir(path_to_folder)==False:
        os.mkdir(path_to_folder)

    np.save( path_to_folder + WOFOST.user_parameters['crop_name']+'_irrigation_ammount', WOFOST.container_of_irrigation_amount)
    np.save( path_to_folder+ WOFOST.user_parameters['crop_name']+'_crop_yields', WOFOST.container_of_mean_yields)
    np.save( path_to_folder+ WOFOST.user_parameters['crop_name']+'_water_loss', WOFOST.container_of_mean_water_loss)


