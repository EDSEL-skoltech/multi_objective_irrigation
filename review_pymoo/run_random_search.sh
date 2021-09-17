#!/bin/bash
for i in {1..10}
do 
	python3 random_search.py --path_to_data_dir '../util/input_data/' --path_to_user_file '../util/input_data/malino_sugar_beet.json' --path_to_CSV_weather '../data/meteo' --num_generation '9000' --path_to_npy_files ./random_review/random_search_$i &
done
