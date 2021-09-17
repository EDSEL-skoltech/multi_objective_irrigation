#!/bin/bash
python3 optimizer_pymoo.py --path_to_data_dir '../util/input_data/' --path_to_user_file '../util/input_data/malino_sugar_beet.json' --path_to_CSV_weather '../data/meteo' --num_generation '300' --path_to_npy_files './review_npy_10/' &
