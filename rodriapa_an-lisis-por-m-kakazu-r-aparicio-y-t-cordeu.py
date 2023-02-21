#!/usr/bin/env python
# coding: utf-8



# Alumnos:
# Mariano Kakazu, 98178
# Rodrigo Aparicio, 98967
# Thomas Cordeu, 99288
# link a repositorio de Github: https://github.com/frisjon/tp1




import numpy as np 
import pandas as pd
import datetime # para convertir a dia de la semana
import calendar # idem

# plots
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




trips = pd.read_csv('../input/trip.csv', low_memory=False)
#Se convierte los dates a datetime64[ns].
trips['start_date'] = pd.to_datetime(trips['start_date'], format='%m/%d/%Y %H:%M')
trips['end_date'] = pd.to_datetime(trips['end_date'], format='%m/%d/%Y %H:%M')




# ejemplo del uso de datetime con día actual
dia_actual = datetime.datetime.today()
dia_actual




# lo paso a dia de la semana
dia_actual.weekday()




# mejor en palabras que en números
calendar.day_name[dia_actual.weekday()]




# función para convertir fecha a día de la semana.
def fecha_a_dia(fecha):
    return calendar.day_name[fecha.weekday()]




#Se crean columnas con los dias de la semana.
trips['start_day_of_week'] = trips['start_date'].map(fecha_a_dia)
trips['end_day_of_week'] = trips['end_date'].map(fecha_a_dia)




trips['horario_inicial'] = trips['start_date'].dt.time
trips['horario_inicial_float'] = trips['start_date'].dt.hour + trips['start_date'].dt.minute / 100




trips['horario_final'] = trips['end_date'].dt.time
trips['horario_final_float'] = trips['end_date'].dt.hour + trips['end_date'].dt.minute / 100




trips['duracion_viaje'] = trips['end_date'] - trips['start_date']




# chequeo tipos
trips.dtypes




# vista final de cómo quedó el dataframe
trips.head()




viajes_no_tomados = trips[(trips['duracion_viaje'] <= '00:03:00') & (trips['start_station_id'] == trips['end_station_id'])]
trips = trips[-((trips['duracion_viaje'] <= '00:03:00') & (trips['start_station_id'] == trips['end_station_id']))]




viajes_no_tomados.id.count()




trips[(trips['duracion_viaje'] > "12:00:00")].head()




viajes_larguisimos = trips[(trips['duracion_viaje'] > "12:00:00") | ((trips['duracion_viaje'] >= "11:00:00") & ((trips['horario_inicial_float'] < 7) | (trips['horario_inicial_float'] > 11)))]
# lo que está después del or es para los viajes que duran entre 11 y 12hs y no empiezan a la mañana
trips = trips[-((trips['duracion_viaje'] > "12:00:00") | ((trips['duracion_viaje'] >= "11:00:00") & ((trips['horario_inicial_float'] < 7) | (trips['horario_inicial_float'] > 11))))]




viajes_larguisimos.id.count()




# ahora los datos quedan más limpios
trips.head()




trips['start_day_of_week'].value_counts().plot(kind='bar', rot=0, figsize=(10,8), color='purple' ,fontsize=13);
plt.title('Cantidad de viajes segun el dia', fontsize=20);
plt.xlabel('Dia de la semana', fontsize=16);
plt.ylabel('Cantidad de viajes', fontsize=20);




dias = trips[['start_day_of_week']]
dias_semana = dias[-(dias['start_day_of_week'] == "Saturday")]
dias_semana = dias[-(dias['start_day_of_week'] == "Sunday")]
dias_finde = dias[(dias['start_day_of_week'] == "Saturday") | (dias['start_day_of_week'] == "Sunday")]




sizes = [dias_semana.start_day_of_week.count(), dias_finde.start_day_of_week.count()]
nombres = ['Dias habiles', 'Fin de semana']

plt.figure(figsize=(6, 6))
plt.title('Distribucion semanal del uso del servicio', fontsize=20)
plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['red', 'yellow'], explode=(0.1, 0))
plt.show()




semana_entera = trips[['start_day_of_week','horario_inicial_float', 'start_station_name', 'end_station_name']].round()
semana_entera['horario_inicial_float'] = semana_entera['horario_inicial_float'].map(lambda x: x if x != 24 else 0)
# 24hs = 0hs
semana_entera.head()




semana = semana_entera[-(semana_entera['start_day_of_week'] == "Saturday")]
semana = semana_entera[-(semana_entera['start_day_of_week'] == "Sunday")]




semana['apariciones'] = semana['start_day_of_week'].map(lambda x: 1) # seteo todas las rows con 1 para despues agrupar
horarios_semana = semana[['horario_inicial_float', 'apariciones']]
semana = semana.drop('apariciones', 1) # vuelvo a dejar el dt como antes
horarios_semana_contador = horarios_semana.groupby('horario_inicial_float').aggregate(sum)




horarios_semana_contador.plot.bar(rot=0, figsize=(10,10), color='green', fontsize=10);
plt.ylabel('Cantidad de viajes', fontsize=20)
plt.xlabel('Horario de inicio del viaje', fontsize=16)
plt.title('Uso del servicio en dias habiles segun el horario', fontsize=17)
plt.legend('')
plt.show()




finde = semana_entera[(semana_entera['start_day_of_week'] == "Saturday") | (semana_entera['start_day_of_week'] == "Sunday")]




finde['apariciones'] = finde['start_day_of_week'].map(lambda x: 1) # seteo todas las rows con 1 para despues agrupar
horarios_finde = finde[['horario_inicial_float', 'apariciones']]
finde = finde.drop('apariciones', 1) # vuelvo a dejar el dt como antes
horarios_finde_contador = horarios_finde.groupby('horario_inicial_float').aggregate(sum)




horarios_finde_contador.plot.bar(rot=0, figsize=(10,10), color='green', fontsize=10);
plt.ylabel('Cantidad de viajes', fontsize=20)
plt.xlabel('Horario de inicio del viaje', fontsize=16)
plt.title('Uso del servicio en el fin de semana segun el horario', fontsize=17)
plt.legend('')
plt.show()




viajes_en_hora_pico_finde = finde[((finde['horario_inicial_float'] >= 11) & (finde['horario_inicial_float'] <= 16))]
viajes_finde_noche = finde[((finde['horario_inicial_float'] >= 20) & (finde['horario_inicial_float'] < 24))]




sizes = [viajes_en_hora_pico_finde.start_day_of_week.count(), viajes_finde_noche.start_day_of_week.count()]
nombres = ['Hora pico', 'Noche']

plt.figure(figsize=(6, 6))
plt.title('Distribucion del uso del servicio los fines de semana', fontsize=20)
plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['orange', 'violet'], explode=(0.1, 0))
plt.show()




viajes_en_hora_pico_semana = semana[((semana['horario_inicial_float'] >= 8) & (semana['horario_inicial_float'] <= 9)) 
                                   | ((semana['horario_inicial_float'] >= 17) & (semana['horario_inicial_float'] <= 18))]




destinos_mas_populares_hora_pico_semana = viajes_en_hora_pico_semana['end_station_name'].value_counts().sort_values(ascending=False)
destinos_mas_populares_hora_pico_semana = destinos_mas_populares_hora_pico_semana.head(10)




destinos_mas_populares_hora_pico_semana.plot(kind='bar', rot=70, figsize=(13,10), color='black', fontsize=10, grid=False);
plt.title('Destinos mas populares en hora pico dia habil', fontsize=20);
plt.ylabel('Cantidad de viajes', fontsize=17);




# función que dado un dataframe con un campo 'start_station_name' y otro 'end_station_name',
# devuelve un diccionario con los start_station como clave y como valor un diccionario con clave el end_station
# y valor la cantidad de viajes de ese trayecto. También devuelve una lista con el trayecto con mayor cantidad
# de viajes junto con el start y end station del mismo. Orden = O(n) siendo n la cantidad de rows del dataframe.
def contador_viajes(dataframe):
    cont_viajes = {}
    viaje_mas_popular = []
    viaje_mas_popular.append(0)
    viaje_mas_popular.append("")
    viaje_mas_popular.append("")
    for index,row in dataframe.iterrows():
        
        if row['start_station_name'] not in cont_viajes:
            cont_viajes[row['start_station_name']] = {}
           
        if row['end_station_name'] not in cont_viajes[row['start_station_name']]:
            cont_viajes[row['start_station_name']][row['end_station_name']] = 1
        else:
            cont_viajes[row['start_station_name']][row['end_station_name']] += 1
        
        if cont_viajes[row['start_station_name']][row['end_station_name']] > viaje_mas_popular[0]:
            viaje_mas_popular[0] = cont_viajes[row['start_station_name']][row['end_station_name']]
            viaje_mas_popular[1] = row['start_station_name']
            viaje_mas_popular[2] = row['end_station_name']
    
    return cont_viajes,viaje_mas_popular




contador_de_viajes_hora_pico_semana,viaje_mas_popular_hora_pico_semana = contador_viajes(viajes_en_hora_pico_semana)
viaje_mas_popular_hora_pico_semana




destinos_mas_populares_hora_pico_finde = viajes_en_hora_pico_finde['end_station_name'].value_counts().sort_values(ascending=False)
destinos_mas_populares_hora_pico_finde = destinos_mas_populares_hora_pico_finde.head(10)




destinos_mas_populares_hora_pico_finde.plot(kind='bar', rot=70, figsize=(13,10), color='black', fontsize=10, grid=False);
plt.title('Destinos mas populares en hora pico fin de semana', fontsize=20);
plt.ylabel('Cantidad de viajes', fontsize=17);




contador_de_viajes_hora_pico_finde,viaje_mas_popular_hora_pico_finde = contador_viajes(viajes_en_hora_pico_finde)
viaje_mas_popular_hora_pico_finde




viernes_y_sab = semana_entera[(semana_entera['start_day_of_week'] == "Saturday") | (semana_entera['start_day_of_week'] == "Friday")]
viernes_y_sab_noche = viernes_y_sab[(viernes_y_sab['horario_inicial_float'] >= 20) & (viernes_y_sab['horario_inicial_float'] < 24)]




destinos_mas_populares_viernes_y_sab_noche = viernes_y_sab_noche['end_station_name'].value_counts().sort_values(ascending=False)
destinos_mas_populares_viernes_y_sab_noche = destinos_mas_populares_viernes_y_sab_noche.head(10)




destinos_mas_populares_viernes_y_sab_noche.plot(kind='bar', rot=70, figsize=(13,10), color='black', fontsize=10, grid=False);
plt.title('Destinos mas populares viernes y sabado por la noche', fontsize=20);
plt.ylabel('Cantidad de viajes', fontsize=17);




semana_entera_con_duracion = trips[['start_day_of_week','horario_inicial_float', 'start_station_name', 'end_station_name','duracion_viaje','duration']].round()
semana_entera_con_duracion.head()




semana_con_duracion = semana_entera_con_duracion[-(semana_entera_con_duracion['start_day_of_week'] == "Saturday")]
semana_con_duracion = semana_entera_con_duracion[-(semana_entera_con_duracion['start_day_of_week'] == "Sunday")]




finde_con_duracion = semana_entera_con_duracion[(semana_entera_con_duracion['start_day_of_week'] == "Saturday") | (semana_entera_con_duracion['start_day_of_week'] == "Sunday")]




semana_con_duracion.duration.mean() / 60 # resultado en minutos




finde_con_duracion.duration.mean() / 60 # resultado en minutos




semana_con_duracion[semana_con_duracion['duracion_viaje'] > "02:00:00"].start_day_of_week.count()




finde_con_duracion[finde_con_duracion['duracion_viaje'] > "02:00:00"].start_day_of_week.count()




finde_viajes_largos = finde_con_duracion[finde_con_duracion['duracion_viaje'] > "00:30:00"]




contador_de_viajes_largos_finde,viaje_largo_mas_popular_finde = contador_viajes(finde_viajes_largos)
viaje_largo_mas_popular_finde




harry_harry = finde_viajes_largos[(finde_viajes_largos['start_station_name'] == 'Harry Bridges Plaza (Ferry Building)') & (finde_viajes_largos['end_station_name'] == 'Harry Bridges Plaza (Ferry Building)')]
harry_harry.describe()




semana_viajes_cortos = semana_con_duracion[(semana_con_duracion['duracion_viaje'] > "00:08:00") & (semana_con_duracion['duracion_viaje'] < "00:20:00")]




contador_de_viajes_cortos_semana,viaje_corto_mas_popular_semana = contador_viajes(semana_viajes_cortos)
viaje_corto_mas_popular_semana




steuart_caltrain = semana_viajes_cortos[(semana_viajes_cortos['start_station_name'] == 'Steuart at Market') & (semana_viajes_cortos['end_station_name'] == 'San Francisco Caltrain (Townsend at 4th)')]
steuart_caltrain.describe()




# funciones para operar con un formato fecha (anio-mes-dia hora:minutos:segundos)
def obtener_dia(fecha):
    return fecha.day
def obtener_mes(fecha):
    return fecha.month
def fecha_sin_hora(fecha):
    return (str(fecha.year) + "-" + str(fecha.month) + "-" + str(fecha.day))




anio_2014 = trips[['start_date','start_day_of_week','horario_inicial_float', 'start_station_name', 'end_station_name','duracion_viaje', 'duration','subscription_type']].round()
anio_2014['horario_inicial_float'] = anio_2014['horario_inicial_float'].map(lambda x: x if x != 24 else 0)
anio_2014 = anio_2014[(anio_2014['start_date'].dt.year) == 2014]
anio_2014['fecha_sin_horario'] = anio_2014['start_date'].map(fecha_sin_hora)
anio_2014['fecha_sin_horario'] = pd.to_datetime(anio_2014['fecha_sin_horario'])
anio_2014['dia'] = anio_2014['start_date'].map(obtener_dia)
anio_2014['mes'] = anio_2014['start_date'].map(obtener_mes)
anio_2014 = anio_2014.sort_values(by='start_date')
anio_2014.head()




anio_2014['viaje'] = anio_2014['start_day_of_week'].map(lambda x: 1) # seteo todas las rows con 1 para despues agrupar
viajes_segun_dia = anio_2014[['fecha_sin_horario', 'viaje']]
anio_2014 = anio_2014.drop('viaje', 1) # vuelvo a dejar el dt como antes
viajes_segun_dia_contador = viajes_segun_dia.groupby('fecha_sin_horario').aggregate(sum)
viajes_segun_dia_contador.head()




viajes_segun_dia_contador.plot.line(figsize=(15,10), color='darkorange', fontsize=15);
plt.xlabel('Meses', fontsize=18)
plt.ylabel('Cantidad de viajes', fontsize=20)
plt.title('Viajes segun transurre el 2014', fontsize=20);
plt.grid(True)
plt.legend('');
plt.show()




viajes_segun_dia_contador[viajes_segun_dia_contador['viaje'] > 1400]




viajes_segun_dia_contador[viajes_segun_dia_contador['viaje'] < 200]




halloween = anio_2014[anio_2014['fecha_sin_horario'] == "2014-10-29"].subscription_type.value_counts()




sizes = [halloween.Subscriber, halloween.Customer]
nombres = ['Suscriptor', 'Cliente']

plt.figure(figsize=(6, 6))
plt.title('Tipos de suscripciones en los viajes durante Halloween', fontsize=20)
plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['pink', 'lightblue'], explode=(0.1, 0))
plt.show()




junio = anio_2014[anio_2014['mes'] == 6]
suscripciones_junio = junio.subscription_type.value_counts()
dia_promedio_junio = suscripciones_junio / 30




sizes = [dia_promedio_junio.Subscriber, dia_promedio_junio.Customer]
nombres = ['Suscriptor', 'Cliente']

plt.figure(figsize=(6, 6))
plt.title('Tipos de suscripciones en los viajes en dia promedio junio', fontsize=20)
plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['pink', 'lightblue'], explode=(0.1, 0))
plt.show()




halloween




dia_promedio_junio




trips = pd.read_csv('../input/trip.csv', low_memory=False)
weather = pd.read_csv('../input/weather.csv', low_memory=False)
station = pd.read_csv('../input/station.csv', low_memory=False)
#Se convierte los dates a datetime64[ns].
trips['start_date'] = pd.to_datetime(trips['start_date'])
weather['date'] = pd.to_datetime(weather['date'])




#Se agrega una nueva columna date que coincide con weather.
trips['date'] = trips['start_date'].apply(lambda x: x.date())
#Se convierte date a datetime64[ns].
trips['date'] = pd.to_datetime(trips['date'])




#Formula para convertir F a C.
def f_to_c(f_temp):
    return round((f_temp - 32) / 1.8, 2)




#Se crean columnas con las temperaturas en C.
weather['max_temperature_c'] = weather['max_temperature_f'].map(f_to_c)
weather['mean_temperature_c'] = weather['mean_temperature_f'].map(f_to_c)
weather['min_temperature_c'] = weather['min_temperature_f'].map(f_to_c)




#Se crean columnas con visibilidad en Km.
weather['max_visibility_km'] = weather['max_visibility_miles'].map(lambda x: x * 1.6)
weather['mean_visibility_km'] = weather['mean_visibility_miles'].map(lambda x: x * 1.6)
weather['min_visibility_km'] = weather['min_visibility_miles'].map(lambda x: x * 1.6)




#Funcion para convertir la duracion de segundos a minutos.
def s_to_m(time):
    return (time / 60)
#Funcion para convertir la duracion de segundos a horas redondeo a 3 decimales.
def s_to_h(time):
    return round((time / 3600),3)




#Se crea una columna con la duracion en minutos y la duracion en horas.
trips['duration_m'] = trips['duration'].map(s_to_m)
trips['duration_h'] = trips['duration'].map(s_to_h)




#Funcion para clasificar estaciones climaticas.
def estacion(date):
    if date.month >= 3 and date.month <= 5:
        return 'Primavera'
    elif date.month >= 6 and date.month <= 8:
        return 'Verano'
    elif date.month >= 9 and date.month <= 11:
        return 'Otoño'
    else:
        return 'Invierno'




#Se crea la columna con la estacion climatica.
trips['estacion_clima'] = trips['date'].map(estacion)




#Se filtra lo mecionado anteriormente,
# las duraciones menores o iguales a 3 minutos con la misma estacion de salida y llegada,
# y los viajes de mas de 12 horas (12 * 3600 = 43200 segundos) y los de entre 11 y 12hs que no empiezan
# a la mañana
trips = trips[-((trips['duration_m'] <= 3.0) & (trips['start_station_id'] == trips['end_station_id']))]
trips['start_hour'] = trips['start_date'].map(lambda x: x.hour)
trips = trips[-((trips['duration'] > 43200) | ((trips['duration'] > 39600) & ((trips['start_hour'] < 7) | (trips['start_hour'] > 11))))]




#Se hace join de trips y station para obtener la ciudad en la cual comenzo el viaje.
#Se renombra el id de trips a id_trip.
trips.rename(columns={'id':'id_trip'}, inplace=True)
station_aux = station[['id', 'city']]
joined_trips_station = trips.merge(station, left_on=['start_station_id'], right_on=['id'])




#Funcion para clasificar la ciudad dependiendo del zipcode.
#La clasificacion se basa en los zip_codes obtenidos en https://www.unitedstateszipcodes.org/.
def zip_ciudad(zip_code):
    if zip_code == 95113:
        return 'San Jose'
    elif zip_code == 94301:
        return 'Palo Alto'
    elif zip_code == 94107:
        return 'San Francisco'
    elif zip_code == 94063:
        return 'Redwood City'
    else:
        return 'Mountain View'




#Se crea una columna city en weather para que coindida con joined_trips_station.
weather['city'] = weather['zip_code'].map(zip_ciudad)




#Se mergean los DataFrames Weather y joined_trips_station en uno solo.
joined = joined_trips_station.merge(weather, left_on=['date', 'city'], right_on=['date', 'city'])




joined.hist(column='mean_temperature_c', grid=True, figsize=(10,10), xrot=90, xlabelsize=15, ylabelsize=20);
plt.xticks(range(10,24,1));
plt.xlabel('Temperatura Promedio(C)', fontsize=15);
plt.ylabel('Cantidad de Viajes', fontsize=20)
plt.title('Analisis de la temperatura promedio', fontsize=20);




joined.hist(column='max_temperature_c', grid=True, figsize=(10,10), xrot=90, xlabelsize=15, ylabelsize=20);
plt.xticks(range(15,25,1));
plt.xlabel('Temperatura Maxima(C)', fontsize=15);
plt.ylabel('Cantidad de Viajes', fontsize=20)
plt.title('Analisis de la temperatura maxima', fontsize=20);




#Funciones para clasificar.
def f_st(row):
    if row['event'] == 'start_date':
        val = 1
    else:
        val = 0
    return val

def f_en(row):
    if row['event'] == 'end_date':
        val = 1
    else:
        val = 0
    return val




trips_station_aux = joined_trips_station[['id_trip', 'start_date', 'end_date', 'city']]
trips_station_melt = pd.melt(trips_station_aux, id_vars=['id_trip','city'], value_vars=['start_date', 'end_date'], var_name='event', value_name='time')
trips_station_melt['time'] = pd.to_datetime(trips_station_melt['time'])




#Se obtiene la cantidad de bicicletas en uso al mismo tiempo.
trips_station_ord = trips_station_melt.sort_values('time', ascending=True) 
trips_station_ord['start_counter'] = trips_station_ord.apply(f_st, axis=1)
trips_station_ord['end_counter'] = trips_station_ord.apply(f_en, axis=1)
trips_station_ord['start'] = trips_station_ord['start_counter'].cumsum()
trips_station_ord['end'] = trips_station_ord['end_counter'].cumsum()
trips_station_ord = trips_station_ord[['id_trip', 'city', 'time', 'start', 'end']]
trips_station_ord['in_use'] = trips_station_ord['start'] - trips_station_ord['end']
trips_station_ord = trips_station_ord.sort_values('in_use', ascending=False)




#Se eliminan los horarios para coincidir con weather.csv.
trips_station_ord['time'] = trips_station_ord['time'].apply(lambda x: x.date())
#Se convierte time a datetime64[ns].
trips_station_ord['time'] = pd.to_datetime(trips_station_ord['time'])




#Se combinan los Dataframes.
joined_simul = trips_station_ord.merge(weather, left_on=['time', 'city'], right_on=['date', 'city'])




#Solo hay que quedarse con el maximo de bicicletas simultaneas para ese dia.
joined_max_simul = joined_simul.drop_duplicates(subset=['time'], keep='first')




#Nos quedamos con los 10 valores maximos y las columnas que interesan.
joined_max_simul_bar = joined_max_simul[:10]
joined_max_simul_bar = joined_max_simul_bar[['time', 'mean_temperature_c', 'max_temperature_c']]
joined_max_simul_bar.set_index('time', inplace=True)




bar = joined_max_simul_bar.plot.bar(figsize=(10,10), fontsize=20);
#Elimina el 00:00:00 del plot.
bar.set_xticklabels(joined_max_simul_bar.index.format());
plt.yticks(range(0,33,2));
plt.xlabel('Fecha(YYYY-MM-DD)', fontsize=15);
plt.ylabel('Temperatura(C)', fontsize=20);
plt.title('Dias con Mayor Uso Simultaneo y sus Temperaturas', fontsize=20);
plt.legend(['Temperatura Promedio(C)', 'Temperatura Maxima(C)'], fontsize=15);




joined['mean_visibility_km'].value_counts(sort=True).plot.bar(figsize=(10,10), fontsize=20);
plt.xlabel('Visibilidad Promedio(km)', fontsize=15);
plt.ylabel('Cantidad de Viajes', fontsize=20)
plt.title('Analisis de Visibilidad Promedio', fontsize=20);




joined['min_visibility_km'].value_counts(sort=True).plot.bar(figsize=(10,10), fontsize=20);
plt.xlabel('Visibilidad Minima(km)', fontsize=15);
plt.ylabel('Cantidad de Viajes', fontsize=20)
plt.title('Analisis de Visibilidad Minima', fontsize=20);




joined['max_visibility_km'].value_counts(sort=True).plot.bar(figsize=(10,10), fontsize=20);
plt.xlabel('Visibilidad Maxima(km)', fontsize=15);
plt.ylabel('Cantidad de Viajes', fontsize=20)
plt.title('Analisis de Visibilidad Maxima', fontsize=20);




#Nos quedamos con los 10 valores maximos y las columnas que interesan.
joined_max_simul_vis_bar = joined_max_simul[:10]
joined_max_simul_vis_bar = joined_max_simul_vis_bar[['time', 'mean_visibility_km', 'min_visibility_km', 'max_visibility_km']]
joined_max_simul_vis_bar.set_index('time', inplace=True)




bar = joined_max_simul_vis_bar.plot.bar(figsize=(10,10), fontsize=20);
#Elimina el 00:00:00 del plot.
bar.set_xticklabels(joined_max_simul_vis_bar.index.format());
plt.yticks(range(0,22,2));
plt.xlabel('Fecha(YYYY-MM-DD)', fontsize=15);
plt.ylabel('Visibilidad(km)', fontsize=20);
plt.title('Dias con Mayor Uso Simultaneo y sus Visibilidades', fontsize=20);
plt.legend(['Visibilidad Promedio(km)', 'Visibilidad Minima(km)', 'Visibilidad Maxima(km)'], fontsize=15);




trips['estacion_clima'].value_counts(sort=True).plot.bar(figsize=(10,10), rot=0, fontsize=20);
plt.yticks(range(0,200000,10000))
plt.xlabel('Estacion Climatica', fontsize=20);
plt.ylabel('Cantidad de Viajes', fontsize=20)
plt.title('Cantidad de Viajes segun la Estacion Climatica', fontsize=20);




grouped_season = trips[['estacion_clima', 'duration_m']].groupby('estacion_clima').aggregate('mean')




grouped_season.plot.bar(figsize=(10,10), rot=0, fontsize=20);
plt.yticks(range(0,21,1));
plt.xlabel('Estacion Climatica', fontsize=20);
plt.ylabel('Duracion(m)', fontsize=20)
plt.title('Duracion promedio de los Viajes segun la Estacion Climatica', fontsize=20);
plt.legend('');




#Se obtienen los 5 viajes con mayor duracion (que poseen datos climaticos) menores a 12 horas
#que ocurran durante el dia (se toman solo viajes que comiencen a la mañana).
#top_dur = joined[(joined['start_hour'] >= 7)  & (joined['start_hour'] <= 11) & (joined['duration_h'] <= 12)]
top_dur = joined[(joined['start_hour'] >= 7)  & (joined['start_hour'] <= 11)]
top_dur = top_dur.sort_values('duration_h', ascending=False)[:5]




top_dur_temp = top_dur[['duration_h', 'max_temperature_c', 'min_temperature_c', 'mean_temperature_c']]
top_dur_temp.set_index('duration_h', inplace=True)




top_dur_temp_bar = top_dur_temp.plot.bar(figsize=(10,10), fontsize=20, rot=0);
plt.yticks(range(0,30,2));
plt.xlabel('Duracion(h)', fontsize=15);
plt.ylabel('Temperatura(C)', fontsize=20);
plt.title('Mayores Duraciones y sus Temperaturas', fontsize=20);
plt.legend(['Temperatura Maxima(C)', 'Temperatura Minima(C)', 'Temperatura Promedio(C)'], fontsize=15);




top_dur_vis = top_dur[['duration_h', 'max_visibility_km', 'min_visibility_km', 'mean_visibility_km']]
top_dur_vis.set_index('duration_h', inplace=True)




top_dur_vis_bar = top_dur_vis.plot.bar(figsize=(10,10), fontsize=20, rot=0);
plt.yticks(range(0,22,2));
plt.xlabel('Duracion(h)', fontsize=15);
plt.ylabel('Visibilidad(km)', fontsize=20);
plt.title('Mayores Duraciones y sus Visibilidades', fontsize=20);
plt.legend(['Visibilidad Maxima(km)', 'Visibilidad Minima(km)', 'Visibilidad Promedio(km)'], fontsize=15);




#Se muestran las duracion(en horas) y su respectiva estacion.
top_dur[['duration_h', 'estacion_clima']]




#Funcion para clasificar lluvia.
def lluvia_y_n(event):
    if isinstance(event, float):
        return 'No'
    elif 'rain' in event.lower():
        return 'Si'
    else:
        return 'No'




joined['lluvia'] = joined['events'].map(lluvia_y_n)




joined['lluvia'].value_counts().plot.bar(figsize=(10,10), fontsize=20, rot=0);
plt.xlabel('Llovio?', fontsize=20);
plt.ylabel('Cantidad de Viajes', fontsize=20)
plt.title('Cantidad de Viajes y la Lluvia', fontsize=20);
plt.legend('');




joined[['lluvia', 'duration_m']].groupby('lluvia').aggregate('mean').plot.bar(figsize=(10,10), fontsize=20, rot=0);
plt.yticks(range(0,21,1));
plt.xlabel('Llovio?', fontsize=20);
plt.ylabel('Duracion(m)', fontsize=20)
plt.title('Duracion promedio de los Viajes y la Lluvia', fontsize=20);
plt.legend('');




from mpl_toolkits.basemap import Basemap

stations = pd.read_csv('../input/station.csv', low_memory=False)

stations.rename(columns={'installation_date':'date', 'long':'lon'}, inplace=True)

#Se cambia el formato de las fechas
stations['date'] = pd.to_datetime(stations['date'])




trips = pd.read_csv('../input/trip.csv', low_memory=False)

#Se eliminan los zip-codes ya que no seran relevantes para estos analisis
trips.drop('zip_code', 1, inplace=True)

trips.rename(columns={'start_date'        :'s_date' , 
                      'end_date'          :'e_date' , 
                      'start_station_name':'ss_name', 
                      'start_station_id'  :'ss_id'  , 
                      'end_station_name'  :'es_name', 
                      'end_station_id'    :'es_id'  , 
                      'subscription_type' :'subs'
                     }, inplace=True)

#Se cambia el formato de las fechas y tiempos de incio y fin de cada viaje
trips['s_date'] = pd.to_datetime(trips['s_date'], format='%m/%d/%Y %H:%M')
trips['e_date'] = pd.to_datetime(trips['e_date'], format='%m/%d/%Y %H:%M')

#Se filtra lo mecionado anteriormente,
# las duraciones menores o iguales a 3 minutos con la misma estacion de salida y llegada
trips = trips[-((trips['duration'] <= 180) & (trips['ss_id'] == trips['es_id']))]
# los viajes de mas de 12 horas (12 * 3600 = 43200 segundos) y los de entre 11 y 12hs que no empiezan
# a la mañana
trips['start_hour'] = trips['s_date'].map(lambda x: x.hour)
trips = trips[-((trips['duration'] > 43200) | ((trips['duration'] > 39600) & ((trips['start_hour'] < 7) | (trips['start_hour'] > 11))))]

#Se ordenan los viajes por id
trips = trips.sort_values(by='id')




#Se separan los tipos de suscripción de cada viaje y se juntan los del mismo tipo
subs = trips.groupby('subs').count()[['id']]
lbls = subs.index.values
vals = subs.values

plt.figure(figsize=(6, 6));
plt.title('Suscriptores vs Clientes', fontsize=20);
plt.pie(vals, explode=(0.1, 0), labels=lbls, colors=['lightgrey', 'gold'], autopct='%1.1f%%', startangle=-25);
plt.savefig('../img/usuarios_del_servicio.png');
plt.show();




#Se separan los viajes segun el año
trips_2013 = trips['2014' > trips['s_date']]
trips_2014 = trips[('2014' < trips['s_date']) & (trips['s_date'] < '2015')]
trips_2015 = trips['2015' < trips['s_date']]

viaje_anio = [trips_2013, trips_2014, trips_2015]
anios = ['2013', '2014', '2015']

plt.figure(figsize=(15, 9));

for i in range(len(viaje_anio)):
    #Se calcula la suma de bicicletas que partieron de las estaciones usadas en cada año
    viaje_anio[i] = viaje_anio[i].groupby('ss_name')['id'].count().sort_values(ascending=False)
    
    ax = plt.subplot(311 + i);
    ax.set_xlim([0, 25500]);
    plt.title('Estaciones mas usadas en ' + anios[i], fontsize=20);
    plt.xlabel('Cantidad de bicicletas totales', fontsize=15);
    plt.ylabel('Estacion de inicio', fontsize=15);
    ax.barh([j for j in range(5)], [valor for valor in viaje_anio[i][:5].values], 
            tick_label=[viaje for viaje in viaje_anio[i][:5].index.values]);
    plt.savefig('../img/estaciones_mas_usadas_en_' + anios[i] + '.png');
plt.tight_layout();




trips_13_15 = trips.groupby('ss_name')['id'].count().sort_values(ascending=False)

plt.figure(figsize=(15, 3));
ax = plt.subplot();
ax.set_xlim([0, 50000]);
plt.title('Estaciones mas usadas', fontsize=20);
plt.xlabel('Cantidad de bicicletas totales', fontsize=15);
plt.ylabel('Estacion de inicio', fontsize=15);
ax.barh([j for j in range(5)], 
        [c for c in trips_13_15[:5].values], 
        tick_label=[c for c in trips_13_15[:5].index.values]);
plt.savefig('../img/estaciones_mas_usadas.png');




from math import pi as PI

def distancia_grados(dist):
    return (180 * dist) / (6371 * PI)

def distancia_km(angulo):
    return angulo * 6371 * PI / 180

def mean(a, b):
    return (a + b) / 2

def mmap(value, min_in, max_in, min_out, max_out):
    m = (max_out - min_out) / float(max_in - min_in)
    b = min_out - m * min_in
    return m * value + b




#Se calcula la cantidad de bicicletas por cada estacion
bicis_por_estacion = trips.groupby('ss_id').count()[['ss_name']].reset_index()
bicis_por_estacion.rename(columns={'ss_id':'id', 'ss_name':'count'}, inplace=True)

#Se agrega el nombre de la estacion, la ciudad en donde queda y sus coordenadas
estaciones = pd.merge(stations.drop(['date', 'dock_count'], 1), bicis_por_estacion, on='id')




#Se separan las estaciones por ciudad
sf = estaciones[estaciones['city'] == 'San Francisco'].reset_index(drop=True)
sj = estaciones[estaciones['city'] == 'San Jose'     ].reset_index(drop=True)
mv = estaciones[estaciones['city'] == 'Mountain View'].reset_index(drop=True)
pa = estaciones[estaciones['city'] == 'Palo Alto'    ].reset_index(drop=True)
rc = estaciones[estaciones['city'] == 'Redwood City' ].reset_index(drop=True)

ciudades = [sf, sj, mv, pa, rc, estaciones]
nombres = ['San Francisco', 'San Jose', 'Mountain View', 'Palo Alto', 'Redwood City', 'Bay Area']
#La cantidad de estaciones se va a utilizar para escalar a los tamaños de las estaciones en la visualizacion
ciudad_cant_estaciones = []
ciudad_bicis = []

#Listas con latitudes y longitudes de cada ciudad
lats = []
lons = []
#Lista con distancias para visualizar en mapas. Las distacias indican en cierto grado, el nivel de zoom
offsets = []

#Para cada ciudad, vemos su cantidad de estaciones y la cantidad total de bicicletas
for ciudad in ciudades:
    ciudad_cant_estaciones.append(ciudad.shape[0])
    ciudad_bicis.append(ciudad['count'].sum())
    #Para mostrar a todas las estaciones, tomamos la media entre la coordenada maxima y minima
    lat_max, lat_min = ciudad['lat'].max(), ciudad['lat'].min()
    lon_max, lon_min = ciudad['lon'].max(), ciudad['lon'].min()
    lats.append(mean(lat_max, lat_min))
    lons.append(mean(lon_max, lon_min))
    offsets.append(max((lat_max - lat_min) * 0.6, (lon_max - lon_min) * 0.6))




plt.figure(figsize=(20, 20))

for i in range(len(ciudades)):
    plt.subplot(331 + i);
    plt.title('Estaciones en ' + nombres[i], fontsize=20);
    mapa=Basemap(projection='merc', resolution='i', epsg=4326, 
                 llcrnrlat=lats[i] - offsets[i], llcrnrlon=lons[i] - offsets[i], 
                 urcrnrlat=lats[i] + offsets[i], urcrnrlon=lons[i] + offsets[i])

    #Aqui se escalan los tamaños de las estaciones dependiendo de la ciudad que se muestre
    mapa.scatter(ciudades[i]['lon'].values, ciudades[i]['lat'].values, marker='o', c='#00ff00', 
                 s=ciudades[i]['count'].apply(lambda x:x/ciudad_cant_estaciones[i]), edgecolors='#00aa00');
    
    mapa.arcgisimage(service='Canvas/World_Light_Gray_Base', xpixels=1000, ypixels=1000);
    plt.savefig('../img/' + nombres[i].lower().replace(' ','_') + '_estaciones_uso.png');




plt.figure(figsize=(6, 6))
plt.title('Concentracion de viajes por ciudad', fontsize=20)
plt.pie(ciudad_bicis[:-1:], labels=nombres[:-1:], autopct='%1.1f%%', explode=(0, .15, .25, .35, .45))
plt.savefig('../img/concentracion_viajes_por_ciudad.png')
plt.show()




#Se calculan las frecuencias de cada trayecto
trayectos_frec = trips[['id', 'ss_name', 'es_name']].groupby(['ss_name', 'es_name'], as_index=False).count()

trayectos_frec.rename(columns={'id':'count'}, inplace=True)




#Dataframe con los nombres de todas las estaciones y sus coordenadas
ss_location = stations[['name', 'lat', 'lon', 'city']]
ss_location.rename(columns={'name':'ss_name', 'lat':'s_lat', 'lon':'s_lon', 'city':'s_city'}, inplace=True)

es_location = stations[['name', 'lat', 'lon', 'city']]
es_location.rename(columns={'name':'es_name', 'lat':'e_lat', 'lon':'e_lon', 'city':'e_city'}, inplace=True)

#Estos dataframes seran combinados con los de trabyectos por la columna que tengan en comun
#Se quiere obtener un dataframe con todos los trayectos, esto es: las estaciones inicial y final, las coordenadas
# de las estaciones y la frecuencia del trayecto




def reducir_trayectos(trayectos_frec):
    l = [] 
        #Lista con indice del dataframe del trayecto eliminado
    for i in range(len(trayectos_frec)):
            #Por cada trayecto
        if i not in l:
            #Si esta en el dataframe
            prim = trayectos_frec.loc[i]
                #Se obtiene el trayecto actual
            for j in range(i, len(trayectos_frec)):
                #Buscamos en lo que resta del dataframe
                if j not in l:
                    #Si el trayecto que queremos ver, esta en el dataframe
                    seg = trayectos_frec.loc[j] 
                        #Se guarda
                    if prim['ss_name'] == seg['es_name'] and prim['es_name'] == seg['ss_name']:
                        #Si coinciden los trayectos
                        trayectos_frec.loc[i]['count'] += trayectos_frec.loc[j]['count']
                            #Se suman las frecuencias
                        trayectos_frec.drop(j, inplace=True)
                            #Se quita al segundo trayecto del dataframe
                        l.append(j)
                            #Se coloca en la lista, para que no halla errores
    return trayectos_frec

#El trayecto A -> B es el mismo que B -> A
#La funcion junta las repeticiones de estos trayectos
tf = reducir_trayectos(trayectos_frec)

#Se añade al dataframe, las ubicaciones de cada estacion
tf = pd.merge(tf, ss_location, on='ss_name')
tf = pd.merge(tf, es_location, on='es_name')




#Este diccionario se va a contener, por ciudad:
#  · lista de tuplas ([lons], [lats], color) que tienen las coord de estaciones de un trayecto y su color
#  · lista de los trayectos de la ciudad
viajes = {'San Francisco':[[], []], 
          'San Jose'     :[[], []], 
          'Redwood City' :[[], []], 
          'Mountain View':[[], []], 
          'Palo Alto'    :[[], []]
         }

for ciudad in viajes:
    #Se filtran los trayectos por ciudad
    #Aqui se quitan los trayectos con estaciones en distintas ciudades
    viajes[ciudad][1] = tf[(tf['s_city'] == ciudad) &                            (tf['e_city'] == ciudad)].drop(['s_city', 'e_city'], 1).reset_index(drop=True)
    
    #Se calcula la cantidad maxima de viajes por cada trayecto en cada ciudad
    #Esto es para luego asignar colores a trayectos segun su frecuencia
    max_val = viajes[ciudad][1]['count'].max()
    for indice in range(len(viajes[ciudad][1])):
        trayecto = viajes[ciudad][1].loc[indice]        
        #Mapeo de la frecuencia del trayecto al intervalo [0;255]
        color = abs(int(mmap(trayecto['count'], 0, max_val, 255, 0)))
        #color = abs(color)
        viajes[ciudad][0].append(([trayecto['s_lon'], trayecto['e_lon']], 
                                  [trayecto['s_lat'], trayecto['e_lat']], color))
        
    #Se ordenan los viajes mas frecuentes al final, asi son mostrados por encima de los trayectos menos frecuentes
    viajes[ciudad][0].sort(key=lambda x:-x[2])




#Ahora visualizamos los trayectos por ciudad
plt.figure(figsize=(20, 12))

for i in range(len(ciudades)-1):
    plt.subplot(231 + i);
    plt.title('Viajes en ' + nombres[i], fontsize=20);
    mapa = Basemap(projection='merc', resolution='i', epsg=4326, 
                   llcrnrlat=lats[i] - offsets[i], llcrnrlon=lons[i] - offsets[i], 
                   urcrnrlat=lats[i] + offsets[i], urcrnrlon=lons[i] + offsets[i])
    
    for trayecto in viajes[nombres[i]][0]:
        #Convertimos el color a un formato RGBA valido
        color = trayecto[2]
        color = '#%02Xff%02X' % (color, color)
        #Ploteamos el trayecto
        mapa.plot(trayecto[0], trayecto[1], 'o-', color=color)

    mapa.arcgisimage(service='Canvas/World_Light_Gray_Base', xpixels=1000, ypixels=1000);
    plt.savefig('../img/' + nombres[i].lower().replace(' ','_') + '_frecuencia_trayectos.png');
    
plt.savefig('../img/frecuencia_trayectos.png');




plt.figure(figsize=(15, 15))
plt.title('Viajes en toda la Bahia', fontsize=20)

offset = distancia_grados(35)
mapa = Basemap(projection='merc', resolution='i', epsg=4326, 
               llcrnrlat=lats[5] - offset, llcrnrlon=lons[5] - offset, 
               urcrnrlat=lats[5] + offset, urcrnrlon=lons[5] + offset )

max_val = tf['count'].max()
tf = tf.sort_values(by='count').reset_index(drop=True)

for indice in range(tf.shape[0]):
    trayecto = tf.loc[indice]
    color = abs(int(mmap(trayecto['count'], 0, max_val, 255, 0)))
    mapa.plot([trayecto['s_lon'], trayecto['e_lon']], 
              [trayecto['s_lat'], trayecto['e_lat']], 
              'o-', color='#%02Xff%02X' % (color, color));

mapa.arcgisimage(service='Canvas/World_Light_Gray_Base', xpixels=1000, ypixels=1000);
plt.savefig('../img/bay_area_frecuencia_trayectos.png');




#Se cuentan los valores para cada id de bicicleta diferente
trips['bike_id'].value_counts().count()




trips_fechas = trips[['s_date', 'bike_id', 'id']].reset_index(drop=True)
trips_fechas['s_date'] = pd.to_datetime(trips_fechas['s_date'].dt.strftime('%Y-%m'))




trips_fechas.groupby(['s_date', 'bike_id'], as_index=False).count()            .groupby('s_date')['bike_id'].count().plot.line(figsize=(20, 5), ylim=(0, 700));
plt.title('Cantidad de bicicletas diferentes por mes', fontsize=20);
plt.xlabel('Meses (2013-08 ~ 2015-08)', fontsize=15);
plt.ylabel('Cantidad de bicicletas utilizadas', fontsize=15);




viajes_anios = trips[['s_date', 'subs']]
_2013 = viajes_anios[(viajes_anios['s_date'].dt.year) == 2013]
_2013 = _2013.sort_values(by='s_date')
_2014 = viajes_anios[(viajes_anios['s_date'].dt.year) == 2014]
_2014 = _2014.sort_values(by='s_date')
_2015 = viajes_anios[(viajes_anios['s_date'].dt.year) == 2015]
_2015 = _2015.sort_values(by='s_date')




_2013.head()




_2013 = _2013[_2013['s_date'].dt.month >= 9] # elimino los pocos datos de agosto que había
_2014 = _2014[_2014['s_date'].dt.month >= 9] 
_2015 = _2015[_2015['s_date'].dt.month >= 5] 




viajes = [_2013.s_date.count(), _2014.s_date.count(), _2015.s_date.count()]
anios = [2013,2014,2015]




d = {'cantidad_viajes': viajes, 'anio': anios}
viajes_por_anio = pd.DataFrame(data=d)
viajes_por_anio = viajes_por_anio.groupby('anio').aggregate(sum)




viajes_por_anio.plot.line(figsize=(15,10), color='violet', fontsize=15);
plt.xlabel('Anios', fontsize=18)
plt.ylabel('Cantidad de viajes', fontsize=20)
plt.title('Cantidad de viajes con el transcurso de los anios', fontsize=20)
plt.grid(True)
plt.legend('');
plt.show()




suscripciones_2013 = _2013.subs.value_counts()
suscripciones_2014 = _2014.subs.value_counts()
suscripciones_2015 = _2015.subs.value_counts()




sizes = [suscripciones_2013.Subscriber, suscripciones_2013.Customer]
nombres = ['Suscriptor', 'Cliente']

plt.figure(figsize=(6, 6))
plt.title('Tipos de suscripciones en los viajes durante 2013', fontsize=20)
plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['lightgreen', 'lightgray'], explode=(0.1, 0))
plt.show()




sizes = [suscripciones_2014.Subscriber, suscripciones_2014.Customer]
nombres = ['Suscriptor', 'Cliente']

plt.figure(figsize=(6, 6))
plt.title('Tipos de suscripciones en los viajes durante 2014', fontsize=20)
plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['lightgreen', 'lightgray'], explode=(0.1, 0))
plt.show()




sizes = [suscripciones_2015.Subscriber, suscripciones_2015.Customer]
nombres = ['Suscriptor', 'Cliente']

plt.figure(figsize=(6, 6))
plt.title('Tipos de suscripciones en los viajes durante 2015', fontsize=20)
plt.pie(sizes, labels=nombres, autopct='%1.1f%%', startangle=20, colors=['lightgreen', 'lightgray'], explode=(0.1, 0))
plt.show()

