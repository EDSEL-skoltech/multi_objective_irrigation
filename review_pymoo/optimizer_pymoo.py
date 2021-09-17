from main_crop_model_dev import Irrigation

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
        
    def minimize_function_20_years_hpc(self, x):
        """
        Minimize this to define optimal day for irrigation
        """
        inputs_years = np.arange(self.NASA_start_year, self.NASA_start_year+20)
        #dates_irrigation = self.irrigation_dates(x)
        self.irrigation_dates_for_many_years_optim = self.irrigation_dates(x)
        
        pool = multiprocessing.Pool(processes=self.num_process)   
        crop_sim_for_20_years = pool.map(self.crop_hpc_20_years, inputs_years)
        yield_of_crop_sim_for_20_years = [crop_sim_for_20_years[i]['TWSO'][-1] for i in range(len(crop_sim_for_20_years))]

        out = np.mean(yield_of_crop_sim_for_20_years)
        return -out
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

    
    def optimizer(self, year):

        """
        Fun to transform int dates to dt.datetime
        """
        import nevergrad as ng
        self.year=year
        dates_irrigation = self.user_parameters['irrigation_events']
        crop_start = self.user_parameters['crop_start']
        #crop_start = crop_start.replace(crop_start[:4], str(year))
        crop_start = self.year_changer(crop_start,year)
        dates_irrigation = [self.year_changer(obj, year) for obj in dates_irrigation]
        dates_irrigation = [dt.datetime.strptime(day, '%Y-%m-%d') for day in dates_irrigation]
        dates_irrigation_integer = [(day-dt.datetime.strptime(crop_start,'%Y-%m-%d')).days for day in dates_irrigation]

        from concurrent import futures
        max_number_of_days = len(pd.date_range(start=self.user_parameters['crop_start'],end=self.user_parameters['crop_end']))
        instrum = ng.p.Tuple(*(ng.p.Scalar(lower=1, upper=max_number_of_days-10).set_integer_casting()for _ in range(len(dates_irrigation_integer))))
        optimizer = ng.optimizers.PSO(parametrization=instrum, budget=80, num_workers=2)
        # optimizer = ng.optimizers.NGOpt4(parametrization=instrum, budget=50, num_workers=2)

        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(self.minimize_function, executor=executor, batch_mode=False)
        
        self.optimal_dates_irrigation = self.irrigation_dates(recommendation.value)
        
        
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
        best dates and ammounts of water for 20 years
        """
#        import nevergrad as ng
        x_dates = x[:len(self.user_parameters['irrigation_events'])]
        x_ammounts = x[len(self.user_parameters['irrigation_events']):]
        amounts = [float(i) for i in x_ammounts]
        #print('Dates:', x_dates)
        #print('ammounts of irrigation, cm', x_ammounts)
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
    
    
        
    def multioptimizer(self, year):

        """
        Fun to transform int dates to dt.datetime
        """
        import nevergrad as ng
        self.year=year
        dates_irrigation = self.user_parameters['irrigation_events']
        crop_start = self.user_parameters['crop_start']
        #crop_start = crop_start.replace(crop_start[:4], str(year))
        crop_start = self.year_changer(crop_start,year)
        dates_irrigation = [self.year_changer(obj, year) for obj in dates_irrigation]
        dates_irrigation = [dt.datetime.strptime(day, '%Y-%m-%d') for day in dates_irrigation]
        dates_irrigation_integer = [(day-dt.datetime.strptime(crop_start,'%Y-%m-%d')).days for day in dates_irrigation]

        from concurrent import futures
        max_number_of_days = len(pd.date_range(start=self.user_parameters['crop_start'],end=self.user_parameters['crop_end']))
        #instrum = ng.p.Tuple(*(ng.p.Scalar(lower=1, upper=max_number_of_days-10).set_integer_casting()for _ in range(len(dates_irrigation_integer))))
        param = ng.p.Dict(
                dates = ng.p.Tuple(*(ng.p.Scalar(lower=1, upper=max_number_of_days-10).set_integer_casting()for _ in range(len(dates_irrigation_integer)))),
                water=ng.p.Tuple(*(ng.p.Scalar(lower=1, upper=15) for _ in range(len(dates_irrigation_integer))))
                )
        
        optimizer = ng.optimizers.CMA(parametrization=param, budget=200, num_workers=1)
        
        optimizer.tell(ng.p.MultiobjectiveReference(), [20, 20])
        # optimizer = ng.optimizers.NGOpt4(parametrization=instrum, budget=50, num_workers=2)

        #with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
         #   recommendation = optimizer.minimize(self.multiobjetive, executor=executor, batch_mode=False)
        recommendation = optimizer.minimize(self.multiobjective)#, verbosity=2)
        #print("Pareto front:")
        #for param in sorted(optimizer.pareto_front(), key=lambda p: p.losses[0]):
        #    print(f"{param} with losses {param.losses}")
        
        
        print('optimal_values:', recommendation.value)
#        self.optimal_dates_irrigation = self.irrigation_dates(recommendation.value)
    
if __name__ == "__main__":
        
    import pandas as pd
    import numpy as np
    import datetime as dt
    import json
    import argparse
    import multiprocessing
    import matplotlib.pyplot as plt
    import os
    import time
    # from telepyth import TelePythClient

    from pymoo.model.problem import Problem
    from pymoo.model.callback import Callback
    from pymoo.factory import get_sampling, get_crossover, get_mutation
    from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover


    from pymoo.algorithms.so_genetic_algorithm import GA
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.factory import get_crossover, get_mutation, get_sampling
    from pymoo.optimize import minimize



    parser = argparse.ArgumentParser(description='Parser_of_input_data')
    parser.add_argument('--path_to_data_dir', type=str, default="/Users/mikhailgasanov/Documents/GIT/agro_rl/pcse_notebooks/data/soil",help='Path to data with soil, crop and other parameter files', required=True)
    parser.add_argument('--path_to_user_file', type=str, default="input_agro_calendar.json",help='JSON file with user input parameters', required=True)
    parser.add_argument('--path_to_CSV_weather', type=str, default="/Users/mikhailgasanov/Documents/machine_learning/NASA_CSV",help='Path to dir with CSV weather database', required=True)
    parser.add_argument('--plot_name', type=str, default='first_plot.png', help='Resulted plot name')
    parser.add_argument('--num_generation', type=int, default=20, help='Num of generation in genetic evolution')
    parser.add_argument('--path_to_npy_files', type=str, default='./npy_files/', help='Path to resulted npy files')
    args = parser.parse_args()

    start = time.time()
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


    mask = ['int']*len(WOFOST.user_parameters['irrigation_events']) + ['real']*len(WOFOST.user_parameters['irrigation_ammounts'])
    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
    })

    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=3.0),
        "int": get_mutation("int_pm", eta=3.0)
    })


    max_number_of_days = len(pd.date_range(start=WOFOST.user_parameters['crop_start'],end=WOFOST.user_parameters['crop_end']))
    x_lower = ([1]*len(WOFOST.user_parameters['irrigation_events'])+[1]*len(WOFOST.user_parameters['irrigation_ammounts']))
    x_upper = ([max_number_of_days-5]*len(WOFOST.user_parameters['irrigation_events'])+[15]*len(WOFOST.user_parameters['irrigation_ammounts']))
    num_of_var = len(x_lower)

    class MyProblem(Problem):
        
        def __init__(self):
            super().__init__(n_var=num_of_var, n_obj=2, 
                            xl=x_lower, 
                            xu=x_upper,
                            elementwise_evaluation=True)
        
        
        def _evaluate(self, x, out, *args, **kwargs):
            f1, f2 = WOFOST.multiobjective(x)
            
            out['F'] = [f1,f2]
                            
                
    class MyCallback(Callback):

        def __init__(self) -> None:
            super().__init__()
            self.data["best"] = []

        def notify(self, algorithm):
            self.data["best"].append(algorithm.pop.get("F").min())


    problem = MyProblem()

    algorithm = NSGA2(
        pop_size=30,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    num_generation = args.num_generation

    print('Start search for optimal solution!')
    res = minimize(
        problem,
        algorithm,
        ('n_gen', num_generation),
        seed=1,
        callback=MyCallback(),
        verbose=True,
        save_history=True
    )
    # Save data for plots 

    import copy
    mdt = copy.deepcopy(res.F)
    mdt[:,0]=mdt[:,0]*-1

    # Save data for plots 
    path_to_saved_files = args.path_to_npy_files
    np.save(os.path.join(path_to_saved_files, WOFOST.user_parameters['crop_name']+'_irrigation_ammount'), WOFOST.container_of_irrigation_amount)
    np.save(os.path.join(path_to_saved_files, WOFOST.user_parameters['crop_name']+'_water_loss'), WOFOST.container_of_mean_water_loss)
    np.save(os.path.join(path_to_saved_files, WOFOST.user_parameters['crop_name']+'_crop_yields'), WOFOST.container_of_mean_yields)
    np.save(os.path.join(path_to_saved_files, WOFOST.user_parameters['crop_name']+'_function_values_for_paretto'), mdt)
    np.save(os.path.join(path_to_saved_files, WOFOST.user_parameters['crop_name']+'_optimal_solutions'), res.X)



    # from telepyth import TelePythClient
    # tp = TelePythClient()
    # fig = plt.figure(figsize=(8,5))
    # ax = plt.subplot(111)
    # ax.scatter(x=mdt[:,0],y=mdt[:,1])
    # ax.set_title ='Objective space'
    # ax.set_xlabel = 'Yield, t/ha'
    # ax.set_ylabel = 'Water loss, cm'
    # fig.savefig('first_run_opt.png')

    # tp.send_figure(fig, 'hello!')
    # print('Plot and save first step')

    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)
    print('Executed time:', time.time()-start)
