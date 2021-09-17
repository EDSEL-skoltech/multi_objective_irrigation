#!/bin/bash
python3 optimizer_pymoo_potato.py --path_to_data_dir '../util/input_data/' --path_to_user_file '../util/input_data/malino_potato.json' --path_to_CSV_weather '../data/meteo' --num_generation '300' --path_to_npy_files './review_npy_10/' &
