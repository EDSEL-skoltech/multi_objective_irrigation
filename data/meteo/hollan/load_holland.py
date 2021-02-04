def update_csv_NASA_weather_database(path_CSV_dir, latitude_min, latitude_max, longitude_min, longitude_max):
    """
    function for downloading NASA weather and creating csv files in folders for future simulation 

    Input: path, latitude_min, latitude_max, longitude_min, longitude_max
    Output: CSV files in dir 

    """
    import time
    import numpy as np
    import pandas as pd
    from pcse.db import NASAPowerWeatherDataProvider
    from pcse.exceptions import PCSEError
    import os
    # UPDATE: Now we download weather for 0.5*0.5 degree zones EU + Russian EU part
    longitude_array = np.arange(longitude_min,longitude_max,step=0.5)
    latitude_array = np.arange(latitude_min,latitude_max,step=0.5)


    path = path_CSV_dir+'/'
    print(path)

    weather_files = os.listdir()
    i = 0

    for latitude in latitude_array:
        for longitude in longitude_array:
            

            # Check presence of files in directory with weather files

            filename = f'NASA_weather_latitude_{latitude}_longitude_{longitude}.csv'
            if filename in weather_files:
                print("There is weather file for:", latitude, longitude)
                continue
            else:
                print("Download it")


                start_time = time.time()
                #API request to NASA database
                def test_weather(latitude,longitude):
                    n=0
                    weather=None
                    while n<10:
                        n+=1
                        try:
                            weather = NASAPowerWeatherDataProvider(latitude, longitude, force_update=True)
                            info='ok in this region'
                            break
                        except KeyError as e:
                            info = e
                            print('В Америке ночь, сервер NASA сладко спит и не хочет отвечать')
                            time.sleep(60*5)
                        except PCSEError as e:
                            print('Погода в этом регионе снова подвела!')
                            info = e
                            break
                    return weather, info
                # Print done if downloaded
                weather, info = test_weather(latitude, longitude)
                
                # if there is no weather for this region or NASA server slept more than 50 mins -- skip this!
                if weather==None:
                    list_to_add = [latitude, longitude, 'NaN', 'NaN', 'NaN', info]
                    str_to_write = ','.join(map(str, list_to_add))
                    with open(path+'weather_database_new.csv', mode='a') as mdt:
                        mdt.write(str_to_write)
                    print('____DONE_____','latitude',latitude, 'longitude',longitude,'____')
                else:
                    # export pcse.weather format to pandas df
                    df_weather = pd.DataFrame(weather.export())
                    #create full range of dates
                    r = pd.date_range(start=df_weather.DAY.min(), end=df_weather.DAY.max())
                    #extend range of dates
                    full_range_weather = df_weather.set_index('DAY').reindex(r).rename_axis('DAY').reset_index()
                    missing_days = (full_range_weather.isna()).sum().sum()

                    print('num_of_missing_days', missing_days)

                    #fill weather with fill forward method in pandas
                    filled_weather = full_range_weather.fillna(method='ffill', axis=0)
                    ##save as csv file
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
                    x = open(path+filename,"w")
                    x.writelines(text)
                    x.close()


                    filled_weather.to_csv(path+filename, mode='a', header=False, index=False)


                    #add info to weather database and save it to csv
                    list_to_add = [latitude, longitude, missing_days, weather.first_date, weather.last_date, info]
                    str_to_write = ','.join(map(str, list_to_add))
                    #weather_database.iloc[i,:] = list_to_add
                    #i += 1
                    #weather_database.to_csv(path+'weather_database.csv', mode='a')

                    with open(path+'weather_database_new.csv', mode='a') as mdt:
                        mdt.write(str_to_write)
                    print('time in sec', time.time() - start_time)
if __name__ == '__main__':
    import pcse
    import pandas as pd
    import numpy as np
    from pcse.db import NASAPowerWeatherDataProvider
    import time

    path_to_csv_dir = '/home/gasanov_mikchail/home/NASA/hollan'
    lat_min = 51
    lat_max = 53
    long_min = 4
    long_max = 6
    update_csv_NASA_weather_database(path_CSV_dir=path_to_csv_dir, latitude_min=lat_min, latitude_max=lat_max, longitude_min=long_min, longitude_max=long_max)

