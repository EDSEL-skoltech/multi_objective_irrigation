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
    
    
