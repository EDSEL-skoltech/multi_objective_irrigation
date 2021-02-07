#!/bin/bash
python3 multiobjective_optimizer.py --path_to_data_dir './util/input_data/' --path_to_user_file './util/input_data/malino_potato.json' --path_to_CSV_weather './data/meteo' --num_generation '5' --population_size '10' --path_to_npy_files './experiments/test/potato/'
