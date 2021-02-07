 #!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime as dt
import json
import os
import argparse
import multiprocessing
import matplotlib.pyplot as plt

from pymoo.model.problem import Problem
from pymoo.model.callback import Callback
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover


from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize

from util.crop_model_en import Optimization
    
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Parser_of_input_data')
    parser.add_argument('--path_to_data_dir', type=str, default="./",help='Path to data with soil, crop and other parameter files', required=True)
    parser.add_argument('--path_to_user_file', type=str, default="input_agro_calendar.json",help='JSON file with user input parameters', required=True)
    parser.add_argument('--path_to_CSV_weather', type=str, default="./",help='Path to dir with CSV weather database', required=True)
    parser.add_argument('--plot_name', type=str, default='first_plot.png', help='Resulted plot name')
    parser.add_argument('--num_generation', type=int, default=10, help='Num of generation in genetic evolution')
    parser.add_argument('--population_size', type=int, default=20, help='Size of population in generation')
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
        pop_size=args.population_size,
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
    path_to_folder = args.path_to_npy_files

    if os.path.isdir(path_to_folder)==False:
        os.mkdir(path_to_folder)
    np.save(path_to_folder + WOFOST.user_parameters['crop_name']+'_irrigation_ammount', WOFOST.container_of_irrigation_amount)
    np.save(path_to_folder + WOFOST.user_parameters['crop_name']+'_water_loss', WOFOST.container_of_mean_water_loss)
    np.save(path_to_folder + WOFOST.user_parameters['crop_name']+'_crop_yields', WOFOST.container_of_mean_yields)
    np.save(path_to_folder + WOFOST.user_parameters['crop_name']+'_function_values_for_paretto', mdt)
    np.save(path_to_folder + WOFOST.user_parameters['crop_name']+'_optimal_solutions', res.X)




    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)
