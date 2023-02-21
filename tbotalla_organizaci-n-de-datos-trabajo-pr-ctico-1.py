#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
import math

# GMaps: no usado porque kaggle no lo soporta. Ver en 
# Installation:
# jupyter nbextension enable --py --sys-prefix widgetsnbextension
# pip install gmaps
# jupyter nbextension enable --py --sys-prefix gmaps
# Docs: http://jupyter-gmaps.readthedocs.io/en/latest/
#import gmaps
#gmaps.configure(api_key="AIzaSyBqxmliEt0lgmXHCKaODzkfY6MsDrSRvNU") 

get_ipython().run_line_magic('matplotlib', 'inline')

# Inputs segun formato de Kaggle
trips = pd.read_csv('../input/trip.csv')
stations = pd.read_csv('../input/station.csv')
weather = pd.read_csv('../input/weather.csv')




trips.start_date = pd.to_datetime(trips.start_date, format='%m/%d/%Y %H:%M')
trips.end_date = pd.to_datetime(trips.end_date, format='%m/%d/%Y %H:%M')
trips_per_hr = trips.loc[:,['start_date','subscription_type']]

# Cambio el nombre de la columna start_date por "hour"
trips_per_hr.rename(columns={'start_date':'hour'}, inplace=True)

# Me quedo solo con la hora
trips_per_hr.hour = trips_per_hr.hour.dt.hour

trips_per_hr.info()




trips_customers = trips_per_hr.loc[trips_per_hr.subscription_type == 'Customer',['hour']]
trips_subscribers = trips_per_hr.loc[trips_per_hr.subscription_type == 'Subscriber',['hour']]
print('Customer\n',trips_customers.head(),'\n\nSubscriber\n',trips_subscribers.head())




data_subs = trips_subscribers.hour.value_counts().sort_index()
data_cust = trips_customers.hour.value_counts().sort_index()
sns.set(font="DejaVu Sans")
fig, ax = plt.subplots(1,1)
fig.set_size_inches((13,5))
fig.tight_layout()
ax.set_title('Viajes por hora según tipo de subscripción', fontsize=18)
ax.set_ylabel('Viajes', fontsize=15)
ax.set_xlabel('Hora', fontsize=15)
ax.plot(data_subs, label='subscribers')
ax.fill_between(data_subs.index,data_subs.values, alpha=0.6)
ax.plot(data_cust, label='customers')
ax.fill_between(data_cust.index,data_cust.values, alpha=0.8)
ax.grid(True)
ax.autoscale(enable=True, axis='both', tight=True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.legend(fontsize=14)




subscribers_by_days = trips.loc[trips.subscription_type == 'Subscriber','start_date']
customers_by_days = trips.loc[trips.subscription_type == 'Customer','start_date']
subscribers_by_month = trips.loc[trips.subscription_type == 'Subscriber','start_date']
customers_by_month = trips.loc[trips.subscription_type == 'Customer','start_date']

subscribers_by_days = pd.to_datetime(subscribers_by_days, format='%m/%d/%Y %H:%M')
customers_by_days = pd.to_datetime(customers_by_days, format='%m/%d/%Y %H:%M')
subscribers_by_month = pd.to_datetime(subscribers_by_month, format='%m/%d/%Y %H:%M')
customers_by_month = pd.to_datetime(customers_by_month, format='%m/%d/%Y %H:%M')

subscribers_by_days = subscribers_by_days.dt.dayofweek
customers_by_days = customers_by_days.dt.dayofweek
subscribers_by_month = subscribers_by_month.dt.month
customers_by_month = customers_by_month.dt.month

subscribers_by_days = subscribers_by_days.value_counts().sort_index()
customers_by_days = customers_by_days.value_counts().sort_index()
subscribers_by_month = subscribers_by_month.value_counts().sort_index()
customers_by_month = customers_by_month.value_counts().sort_index()

subscribers_by_days.rename('Subscribers', inplace=True)
customers_by_days.rename('Customers', inplace=True)
subscribers_by_month.rename('Subscribers', inplace=True)
customers_by_month.rename('Customers', inplace=True)

df_by_days = pd.concat([subscribers_by_days,customers_by_days], axis=1)
print('by days\n',df_by_days)
df_by_month = pd.concat([subscribers_by_month,customers_by_month], axis=1)
print('by month\n',df_by_month)




days_of_week = ['lunes','martes','miércoles','jueves','viernes','sábado','domingo']
ax2 = df_by_days.plot(kind='bar', figsize=(14,5))
ax2.set_title('Viajes por día y tipo de suscripcíon', fontsize=18)
ax2.set_xlabel('Día', fontsize=16)
ax2.set_ylabel('Viajes (cantidad)', fontsize=16)
ax2.set_xticklabels(days_of_week,rotation='horizontal', fontsize=12)
ax2.legend(prop={'size':14})




months = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre']
ax3 = df_by_month.plot(kind='bar', figsize=(14,5))
ax3.set_title('Viajes por més y tipo de suscripcíon', fontsize=18)
ax3.set_xlabel('Mes', fontsize=16)
ax3.set_ylabel('Viajes (cantidad)', fontsize=16)
ax3.set_xticklabels(months,rotation='horizontal', fontsize=12)
ax3.legend(prop={'size':14})




weather.date = pd.to_datetime(weather.date, format='%m/%d/%Y')
temp_per_month = weather.loc[:,['date','mean_temperature_f']]

# convierto F° a C°
temp_per_month.mean_temperature_f = temp_per_month.mean_temperature_f.apply(lambda x: round((x-32)/1.8, 1))
temp_per_month.rename(columns={'mean_temperature_f':'mean_temp_c'}, inplace=True)

temp_per_month.date = temp_per_month.date.dt.month




fig, ax_cus_trips = plt.subplots(1,1)
ax_temp = ax_cus_trips.twinx()
fig.set_size_inches((15,4))
ax_cus_trips.set_title('INFLUENCIA DE TEMPERATURA EN CUSTOMERS', fontsize=18)
ax_temp.plot(temp_per_month.groupby('date').mean(), label='temp')
ax_temp.legend(loc=1, fontsize=14)
ax_temp.grid(False)
ax_temp.set_ylabel('Temperatura C°', fontsize=14)
ax_cus_trips.bar(df_by_month.index, df_by_month.Customers, color='coral', label='customers')
ax_cus_trips.legend(loc=2, fontsize=14)
ax_cus_trips.set_ylabel('Cantidad viajes', fontsize=14)
ax_cus_trips.set_xlabel('Mes', fontsize=14)
ax_cus_trips.set_xticks(df_by_month.index)
("")




trips.start_date = pd.to_datetime(trips.start_date, format='%m/%d/%Y %H:%M')
trips.end_date = pd.to_datetime(trips.end_date, format='%m/%d/%Y %H:%M')

trips_by_day_and_hour = trips.loc[:,['start_date']]
trips_by_day_and_hour['day'] = trips_by_day_and_hour.start_date.dt.weekday_name
trips_by_day_and_hour['hour'] = trips_by_day_and_hour.start_date.dt.hour
trips_by_day_and_hour = trips_by_day_and_hour.loc[:,['day','hour']]
tmp = trips_by_day_and_hour.groupby('day')
tmp = pd.DataFrame(tmp.hour.value_counts())
tmp.rename(columns={'hour':'count'},inplace=True)
tmp.reset_index(inplace=True)
tmp = tmp.pivot(index='hour', columns='day', values='count')
tmp = tmp[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]




fig, ax = plt.subplots(1,1)
fig.set_size_inches((15.5,5))
#fig.tight_layout()
ax.plot(tmp)
ax.set_title('Viajes por hora según el día', fontsize=18)
ax.legend(tmp.columns, fontsize=14)
ax.autoscale(axis='x', tight=True)
ax.set_xlabel('Hora', fontsize=14)
ax.set_ylabel('Cantidad de viajes', fontsize=14)
ax.set_xticks(tmp.index)
("")




tiempoEnSF = weather[weather['zip_code'] == 94107]

def aCels(fahr):
        return (fahr -32)/1.8
    
temperaturasMedias = tiempoEnSF.loc[:,'mean_temperature_f' ]
print('La temperatura maxima es {0:.1f} °C y la minima es {1:.1f}°C'.format(aCels(temperaturasMedias.max()),aCels(temperaturasMedias.min())))

eventos = tiempoEnSF[tiempoEnSF.loc[:,'events'].notnull()].head()
eventos.head(3)




viajesDias = pd.to_datetime(trips.loc[:,'start_date'], format='%m/%d/%Y %H:%M')
viajes_meses = viajesDias.dt.month
rankingMesesTotal = viajes_meses.value_counts().sort_index()
rankingMesesTotal

mes = [31,28,31,30,31,30,31,31,30,31,30,31]
for i in range(12):
    rankingMesesTotal.iloc[i] = rankingMesesTotal.iloc[i] / mes[i]




rankingMesesTotal




trips.iloc[temperaturasMedias.argmax()]
tiempoDias = pd.to_datetime(tiempoEnSF.loc[:,'date'], format='%m/%d/%Y')

df =  pd.DataFrame(data = temperaturasMedias)
df = df.join(tiempoDias).set_index('date')

viajesPorDia = viajesDias.dt.date.value_counts().sort_index()
df = df.join(viajesPorDia).rename(columns={'start_date' : 'cantidad_viajes', 'mean_temperature_f': 'temperatura_media_f'})




df.plot(subplots=True, layout=(2,1), figsize=(15,8))




numeroSemana = tiempoDias.dt.weekday
semana = (numeroSemana != 5) & (numeroSemana != 6)

semana = semana.reset_index()['date']

dfSemana = df.reset_index()
dfSemana = dfSemana[semana].set_index('date')

dfSemana.head()




dfSemana.plot(subplots=True, layout=(2,1), figsize=(15,8))




diasConPocosViajes = dfSemana[dfSemana['cantidad_viajes'] < 700]
pocosViajes2014 = diasConPocosViajes[(diasConPocosViajes.index.year > 2013) & (diasConPocosViajes.index.year < 2015)]
pocosViajes2014.head()




diasSF = tiempoDias.reset_index()['date']
diasSF.name = 'dia'

tiempoEnSF = weather[weather['zip_code'] == 94107].reset_index()
tiempoEnSF = tiempoEnSF.join(diasSF).set_index('dia')

tiempoEnSF = tiempoEnSF.drop('date', 1)
tiempoEnSF = tiempoEnSF.drop('index', 1)
tiempoEnSF.head()




tiempoEnSF[tiempoEnSF['precipitation_inches'] == 'T']




tiempoDePocosViajes2014 = tiempoEnSF.loc[pocosViajes2014.index]
tiempoEnSF[(tiempoEnSF.index.year > 2013) & (tiempoEnSF.index.year < 2015)]['events'].value_counts()
tiempoDePocosViajes2014['events'].value_counts()
tiempoDePocosViajes2014[tiempoDePocosViajes2014['events'].notnull()].head()




tiempo2014 = tiempoEnSF[(tiempoEnSF.index.year > 2013) & (tiempoEnSF.index.year < 2015)]
eventos2014 = tiempo2014[tiempo2014.events.notnull()]
dfSemana.loc[eventos2014.index].head()




df2 = df.join(tiempoEnSF['cloud_cover'])
df2Semana = df2.loc[semana.values]
df2Semana.plot(subplots=True, layout=(3,1), figsize=(15,9))




ratioNubesViajes = (df2Semana.cloud_cover / df2Semana.cantidad_viajes).sort_values(ascending=False).head(50)
#ratioNubesViajes.head()
df2Semana.loc[ratioNubesViajes.index].head()




plt.title('Viajes vs cantidad de nubes')
plt.xscale('log')
plt.plot(ratioNubesViajes.values , df2Semana.loc[ratioNubesViajes.index]['cantidad_viajes'].values)




column = 'min_temperature_f'
df1 = df.join(tiempoEnSF[column])
dfSemana = df1.loc[semana.values]
maxX = dfSemana[column].max()
maxY = dfSemana['cantidad_viajes'].max()
ratio =    ((dfSemana[column]/maxX) / (dfSemana['cantidad_viajes']/maxY))
ratio.name = 'ratio'
nuevoDf = dfSemana.join(ratio)
nuevoDf.plot(subplots=True, layout=(4,1), figsize=(15,9))




column = 'mean_visibility_miles'
df1 = df.join(tiempoEnSF[column])
dfSemana = df1.loc[semana.values]
ratio =  1 / ( dfSemana[column]  * dfSemana['cantidad_viajes'])
ratio.name = 'ratio'
nuevoDf = dfSemana.join(ratio)
nuevoDf.plot(subplots=True, layout=(4,1), figsize=(15,9))




customerTrips = trips[trips.subscription_type == 'Customer']
subscribersTrips = trips[trips.subscription_type != 'Customer']

viajesCustomersDias = pd.to_datetime(customerTrips.loc[:,'start_date'], format='%m/%d/%Y %H:%M').dt.date.value_counts().sort_index()
viajesSubscribersDias = pd.to_datetime(subscribersTrips.loc[:,'start_date'], format='%m/%d/%Y %H:%M').dt.date.value_counts().sort_index()

tiempoSFCustomers = tiempoEnSF.join(viajesCustomersDias).rename(columns={'start_date':'cantidad_viajes_C'})
tiempoSFSubscribers = tiempoEnSF.join(viajesSubscribersDias).rename(columns={'start_date':'cantidad_viajes_C'})
tiempoSFTotal = tiempoSFCustomers.join(viajesSubscribersDias).rename(columns={'start_date':'cantidad_viajes_S'})




#subscribers
column = 'mean_temperature_f'
nuevoDf = tiempoSFTotal.loc[:,[column,'cantidad_viajes_S']]
ratio1 =  nuevoDf[column]  / ( nuevoDf['cantidad_viajes_S'])
ratio1.name = 'ratioC'
#ratio2 =  nuevoDf[column]  / ( nuevoDf['cantidad_viajes_S'])
#ratio2.name = 'ratioS'
#nuevoDf = nuevoDf.join(ratio2)
nuevoDf.plot(subplots=True, layout=(3,1), figsize=(15,9))




column = 'mean_temperature_f'
nuevoDf = tiempoSFTotal.loc[:,[column, 'cantidad_viajes_S']]
ratio = 1/ (nuevoDf[column]/nuevoDf[column].max()) * (1 / ( nuevoDf['cantidad_viajes_S']/nuevoDf['cantidad_viajes_S'].max()))
ratio.name = 'ratio'
nuevoDf = nuevoDf.join(ratio)
nuevoDf[semana.values].plot(subplots=True, layout=(3,1), figsize=(15,9))




column = 'cloud_cover'
nuevoDf = tiempoSFTotal.loc[:,[column, 'cantidad_viajes_S']]
ratio = (nuevoDf[column]/nuevoDf[column].max()) * (1 / ( nuevoDf['cantidad_viajes_S']/nuevoDf['cantidad_viajes_S'].max()))
ratio.name = 'ratio'
nuevoDf = nuevoDf.join(ratio)
nuevoDf[semana.values].plot(subplots=True, layout=(3,1), figsize=(15,9))




column = 'precipitation_inches'
nuevoDf = tiempoSFTotal.loc[:,['cantidad_viajes_S']]
ratio = (dfSemana[column]/dfSemana[column].max()) * (1 / ( nuevoDf['cantidad_viajes_S']/nuevoDf['cantidad_viajes_S'].max()))
ratio.name = 'ratio'
nuevoDf = nuevoDf.join(ratio)
nuevoDf = nuevoDf.join(dfSemana[column])
nuevoDf[semana.values].plot(subplots=True, layout=(3,1), figsize=(15,9))




column = 'mean_dew_point_f'
nuevoDf = tiempoSFTotal.loc[:,[column, 'cantidad_viajes_S']]
ratio = (1/ (nuevoDf[column]/nuevoDf[column].max())) * (1 / ( nuevoDf['cantidad_viajes_S']/nuevoDf['cantidad_viajes_S'].max()))
ratio.name = 'ratio'
nuevoDf = nuevoDf.join(ratio)
nuevoDf[semana.values].plot(subplots=True, layout=(3,1), figsize=(15,9))




viajesSemana = viajesPorDia[semana.values]
viajesSemana[viajesSemana< 300]
tiempoEnSF.loc['2014-11-27']




column = 'mean_humidity'
nuevoDf = tiempoSFTotal.loc[:,[column, 'cantidad_viajes_S']]
ratio =  (nuevoDf[column]/nuevoDf[column].max()) * (1 / ( nuevoDf['cantidad_viajes_S']/nuevoDf['cantidad_viajes_S'].max()))
ratio.name = 'ratio'
nuevoDf = nuevoDf.join(ratio)
nuevoDf[semana.values].plot(subplots=True, layout=(3,1), figsize=(15,9))




minimasDuraciones = trips['duration'].sort_values(inplace=False).head()
maximasDuraciones = trips['duration'].sort_values(ascending= False,inplace=False).head()
print(minimasDuraciones)
print('---')
print(maximasDuraciones)




trips.loc[maximasDuraciones.index]




tripsCustomer = trips[trips['subscription_type'] == 'Customer']
tiempo = tripsCustomer['duration'].sum()/ len(tripsCustomer)
print('el promedio de duracion de los customers es',tiempo, 's')




tripsSubscribers = trips[trips['subscription_type'] == 'Subscriber']
tiempo = tripsSubscribers['duration'].sum()/ len(tripsSubscribers)
print('el promedio de duracion de los subcribers es',tiempo, 's')




# Plot duraciones promedio customers-subscribers
tiempoPromedioCustomersEnMin = int(round(((tripsCustomer['duration'].sum()/len(tripsCustomer))/60),0))
tiempoPromedioSubscribersEnMin = int(round(((tripsSubscribers['duration'].sum()/len(tripsSubscribers))/60),0))
subscribersCustomersAxis = np.arange(2)
subscribersCustomersDurationsAxis = [tiempoPromedioCustomersEnMin, tiempoPromedioSubscribersEnMin]

fig = plt.figure()
fig.suptitle('Duración de viajes promedio', fontsize = 18)
plt.ylabel('Duración(m)', fontsize = 12)
my_xticks = ['Customers','Subscribers']
plt.xticks(subscribersCustomersAxis, my_xticks, fontsize = 15)

axes = plt.gca()
axes.set_ylim([0,75])

colors = ["#b3f269", "#00d73e"]

plt.bar(subscribersCustomersAxis, subscribersCustomersDurationsAxis, align='center', color=colors)




tripsCustomer = trips[trips['subscription_type'] == 'Customer']
tripsSubscribers = trips[trips['subscription_type'] == 'Subscriber']

tripsDict = {'subscription_type' : (['Customers', 'Subscribers']),
               'trips_amount' : ([len(tripsCustomer), len(tripsSubscribers)])}

tripsDf = pd.DataFrame(tripsDict)
colors = ["#b3f269", "#00d73e"]

plt.pie(
    tripsDf['trips_amount'],
    labels=tripsDf['subscription_type]'],
    shadow=False,
    colors=colors,
    startangle=0,
    autopct='%1.1f%%'
    )

plt.axis('equal')
plt.tight_layout()
plt.show()




start_stations = trips.loc[:,'start_station_name'].value_counts()
print(start_stations.head(3),'\n')
end_stations = trips.loc[:,'end_station_name'].value_counts()
print(end_stations.head(3))




from matplotlib.ticker import FuncFormatter

fig, ax = plt.subplots(1, 2)
fig.set_size_inches((7.5,18))
sns.set(font="DejaVu Sans")
# Gráfico para las estaciones de comienzo
ax[0].barh(np.arange(len(start_stations.index)),start_stations.values,tick_label=start_stations.index)
ax[0].invert_yaxis()
ax[0].grid(True,axis='x')
ax[0].xaxis.set_tick_params(labeltop='on')
ax[0].xaxis.set_label_position(position='top')
ax[0].set_xlabel('Cantidad de viajes comienzo (miles)', fontsize=12, labelpad=15)
ax[0].set_ylabel('Estaciones', fontsize=12)
ax[0].autoscale(tight=True, axis='y')
ax[0].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%g" % (x*1e-3)))
# Gráfico para las estaciones de comienzo
ax[1].barh(np.arange(len(end_stations.index)),end_stations.values,tick_label=end_stations.index)
ax[1].invert_yaxis()
ax[1].invert_xaxis()
ax[1].grid(True,axis='x')
ax[1].xaxis.set_tick_params(labeltop='on')
ax[1].xaxis.set_label_position(position='top')
ax[1].yaxis.set_tick_params(left='off', right='on', labelright='on', labelleft='off')
ax[1].yaxis.set_label_position(position='right')
ax[1].set_xlabel('Cantidad de viajes fin (miles)', fontsize=12, labelpad=15)
ax[1].set_ylabel('Estaciones', fontsize=12)
ax[1].autoscale(tight=True, axis='y')
ax[1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%g" % (x*1e-3)))
# Ajuste de espacio entre subplots
plt.subplots_adjust(wspace=0.05)




# Obtengo el top 10 de estaciones con mayor actividad
top10_start_stations = start_stations.head(10)
top10_end_stations = end_stations.head(10)

# Obtengo las 10 estaciones con menor actividad
last10_start_stations = start_stations.tail(10)
last10_end_stations = end_stations.tail(10)

print('TOP 10 START STATIONS')
top10_start_stations




print('TOP 10 END STATIONS')
top10_end_stations




# Me quedo solo con id, name y dock_count y ordeno por cantidad de docks
top_stations_ordered_by_dock_count = stations.loc[:,['id','name','dock_count']].sort_values(by='dock_count', ascending=False)
print(top_stations_ordered_by_dock_count.head(10)[['name','dock_count']])




last_stations_ordered_by_dock_count = stations.loc[:,['id','name','dock_count']].sort_values(by='dock_count')
print(last_stations_ordered_by_dock_count.head(10)[['name','dock_count']])




iter_csv = pd.read_csv('../input/status.csv', iterator=True, chunksize=5000000)
# Como el set de datos es muy grande como para manejarlo en memoria, lo voy cargando por partes, aplico
# filtros a cada parte según la relevancia de los datos y concateno cada parte a la procesada previamente.
# Se filtran todos los registros fuera del intervalo de 6 a 20 hs porque no presentan actividad significativa.
# Para poder acotar el volumen de datos, en lugar de tomar los registros por cada minuto, tomo registros cada 5 minutos.
# La informacion debería seguir siendo representativa y el volumen de datos manejable.
df = pd.DataFrame()
i = 0
total_size = 0
parts = math.ceil(71984434/5000000) # cant de registros / chunksize
print('filtering...')
for chunk in iter_csv:
    total_size = total_size + len(chunk)
    chunk.time = pd.to_datetime(chunk.time)
    # Filtro horarios con actividad despreciable
    chunk = chunk.loc[(chunk.time.dt.hour>=6)&(chunk.time.dt.hour<=20),]
    # Tomo intervalo de registros cada 5 min en lugar de 1
    chunk = chunk.loc[chunk.time.dt.minute.mod(5) == 0,]
    if(df.empty):
        df = chunk
    else:
        df = df.append(chunk)
    i = i+1
    print('processed ' + str(i) + ' chunks of ' + str(parts))

print('***** finished processing *****')
print('filtered -> ' + str(total_size - len(df)) + ' of ' + str(total_size))
print('-------------------------------------\n')
print(df.info())




new_df = df.reset_index()
new_df = new_df.loc[:,['station_id','bikes_available','time']].rename(columns={'station_id':'id'})

grouped = new_df.loc[:,['id','bikes_available']].groupby('id').mean().reset_index()
grouped.rename(columns={'bikes_available':'prom_bikes_available'},inplace=True)

merged = pd.merge(grouped,stations.loc[:,['id','name','dock_count']], on='id')
merged = merged[['id','name','dock_count','prom_bikes_available']]

merged['availability_%'] = merged.prom_bikes_available*100/merged.dock_count
merged.sort_values(by='availability_%')




# Me fijo el availability_% en horarios pico. Esto es en los intervalos 7-9 y 16-18
rush_hour_status = new_df.loc[((new_df.time.dt.hour >= 6) & (new_df.time.dt.hour <= 8))|((new_df.time.dt.hour >= 16) & (new_df.time.dt.hour <= 18)),]
rush_hour_status = rush_hour_status.loc[:,['id','bikes_available']].groupby('id').mean().reset_index()
rush_hour_status.rename(columns={'bikes_available':'prom_bikes_available'},inplace=True)

rush_hour_status = pd.merge(rush_hour_status, stations.loc[:,['id','name','dock_count']], on='id')
rush_hour_status = rush_hour_status[['id','name','dock_count','prom_bikes_available']]

rush_hour_status['availability_%'] = rush_hour_status.prom_bikes_available*100/merged.dock_count
rush_hour_status.sort_values(by='availability_%')




empty_stations = pd.merge(new_df.loc[new_df.bikes_available == 0,],stations.loc[:,['id','name']], on='id')
empty_stations.time = empty_stations.time.dt.time




empty_stations['time'].mode()
empty_stations.loc[empty_stations.time == empty_stations['time'].mode()[0],]['name'].mode()




empty_stations['name'].mode()
empty_stations.loc[empty_stations.name == empty_stations['name'].mode()[0],]['time'].mode()




empty_stations.name.value_counts().head(10)




# pongo los segundos en 0 para poder agrupar correctamente por time
empty_stations.time = empty_stations.time.apply(lambda x: x.replace(second=0))
# Agrupo segun la hora, del conjunto de estaciones agrupado para cada hora tomo el que más veces se repite
most_frequent_empty_stations = empty_stations.groupby(by='time').agg(lambda x: x.value_counts().index[0])['name']
most_frequent_empty_stations = most_frequent_empty_stations.reset_index()
# Para cada momento del día se muestra la estación que con más frecuencia se queda sin bicicletas
most_frequent_empty_stations




# Para ver las estaciones que primero se agotan, tomo las primeras estaciones no repetidas
# Top 10 estaciones que usualmente se quedan sin bicicletas primero
most_frequent_empty_stations.drop_duplicates(subset='name')




top5_first_empty_stations = pd.DataFrame([x for x in most_frequent_empty_stations.name.unique()]).head(5).rename(columns={0:'name'})
top5_first_empty_stations = pd.merge(top5_first_empty_stations,stations.loc[:,['name','lat','long','dock_count','city']], on='name')
top5_first_empty_stations




markers = []
for row in top5_first_empty_stations.values:
    markers.append({
        'name':row[0],
        'location':(row[1],row[2])
    })
markers




# GMaps comentado para Kaggle
# MAPA
#empty_stations_map = gmaps.Map()
#stationLocations = [station['location'] for station in markers]
#stationInfo = [station['name'] for station in markers]
#stations_layer = gmaps.marker_layer(stationLocations, info_box_content=stationInfo)
#empty_stations_map.add_layer(stations_layer)
#empty_stations_map




# Convert date columns to datetime64
trips['start_date'] = pd.to_datetime(trips['start_date'], format='%m/%d/%Y %H:%M', errors='coerce')
trips['end_date'] = pd.to_datetime(trips['end_date'], format='%m/%d/%Y %H:%M', errors='coerce')
stations['installation_date'] = pd.to_datetime(stations['installation_date'], format='%m/%d/%Y', errors='coerce')
weather['date'] = pd.to_datetime(weather['date'], format='%m/%d/%Y', errors='coerce')




# Estaciones desbalanceadas: ver las estaciones para las cuales los viajes desde esa estacion son mayores que los viajes
# hacia esa estacion, o al reves.
groupStartStations = pd.DataFrame(trips.groupby(['start_station_name']).size().sort_values(ascending=False), columns=['cant_viajes_desde']) # Cuenta ocurrencias de cada grupo
groupEndStations = pd.DataFrame(trips.groupby(['end_station_name']).size().sort_values(ascending=False), columns=['cant_viajes_hasta']) # Cuenta ocurrencias de cada grupo

groupStartStations = groupStartStations.reset_index()
groupEndStations = groupEndStations.reset_index()

grouped = groupStartStations.set_index('start_station_name').join(groupEndStations.set_index('end_station_name'))
grouped = grouped.reset_index()
grouped.index.names = ['station']
grouped.columns = ['station', 'cant_viajes_desde', 'cant_viajes_hasta']

def calcularPorcentajeDesbalance(cant_viajes_desde, cant_viajes_hasta):
    total = cant_viajes_desde + cant_viajes_hasta 
    return (100 * (cant_viajes_desde - cant_viajes_hasta) / total)

unbalancedStationsList = []
for index, row in grouped.iterrows():
    porcentajeDesbalance = calcularPorcentajeDesbalance(row['cant_viajes_desde'], row['cant_viajes_hasta'])
    unbalancedStationsList.append((row['station'], porcentajeDesbalance))

# Estaciones con mas egresos que ingresos
unbalancedStationsListHead10 = sorted(unbalancedStationsList, key=lambda tup: tup[1], reverse=True)[:10]

# Estaciones con mas ingresos que egresos
unbalancedStationsListTail10 = sorted(unbalancedStationsList, key=lambda tup: tup[1], reverse=False)[:10]




# Grafico estaciones con mas egresos que ingresos
x_values = []
y_values = []

for i in unbalancedStationsListHead10:
    x_values.append(i[0])
    y_values.append(i[1])
    
x_pos = np.arange(len(x_values)) 
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12,6))
plt.bar(x_pos, y_values, align='center', color='brown')
axes = plt.gca()
axes.set_ylim([0,30])


plt.xticks(x_pos, x_values, rotation='vertical')
plt.ylabel('% Desbalance')




# GMap comentado para Kaggle
# Ubicacion estaciones con mas egresos que ingresos
unbalancedStationsListHead10
myGmap2 = gmaps.Map()
stationMarkers = []

# Analizo solo la mas desbalanceada
mostUnbalancedStation = stations.loc[stations['name'] == 'Grant Avenue at Columbus Avenue']
stationMarkers.append({"name": 'Grant Avenue at Columbus Avenue', "location": (float(mostUnbalancedStation.lat), float(mostUnbalancedStation.long))})
    
stationLocations = [station["location"] for station in stationMarkers]
infoBoxTemplate = """
<dl>
<dt>Station name</dt><dd>{name}</dd>
</dl>
"""
stationInfo = [infoBoxTemplate.format(**station) for station in stationMarkers]
stations_layer = gmaps.marker_layer(stationLocations, info_box_content=stationInfo)
myGmap2.add_layer(stations_layer)

myGmap2




# Grafico estaciones con mas ingresos que egresos
x_values = []
y_values = []

for i in unbalancedStationsListTail10:
    x_values.append(i[0])
    y_values.append(abs(i[1]))
    
x_pos = np.arange(len(x_values)) 
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12,6))
plt.bar(x_pos, y_values, align='center', color='brown')
axes = plt.gca()
axes.set_ylim([0,20])


plt.xticks(x_pos, x_values, rotation='vertical')
plt.ylabel('% Desbalance')




'''
# Ubicacion estaciones con mas ingresos que egresos
myGmap3 = gmaps.Map()
stationMarkers = []

# Analizo solo la mas desbalanceada
mostUnbalancedStation = stations.loc[stations['name'] == 'Redwood City Medical Center']
stationMarkers.append({"name": 'Redwood City Medical Center', "location": (float(mostUnbalancedStation.lat), float(mostUnbalancedStation.long))})
    
stationLocations = [station["location"] for station in stationMarkers]
infoBoxTemplate = """
<dl>
<dt>Station name</dt><dd>{name}</dd>
</dl>
"""
stationInfo = [infoBoxTemplate.format(**station) for station in stationMarkers]
stations_layer = gmaps.marker_layer(stationLocations, info_box_content=stationInfo)
myGmap3.add_layer(stations_layer)

myGmap3
'''




mostCommonTrips = trips.groupby(['start_station_name','end_station_name']).size().reset_index()
mostCommonTrips.rename(columns={0 : 'count'}, inplace=True)
mostCommonTrips = mostCommonTrips.sort_values(['count'], ascending=False)

mostCommonTrips = mostCommonTrips.head(10)
mostCommonTrips




# Grafico top 10 rutas mas frecuentes
rutas = []
cantViajes = []
for index, row in mostCommonTrips.iterrows():
    ruta = str(row['start_station_name']) + " - " + str(row['end_station_name']) 
    rutas.append(ruta)
    cantViajes.append(row['count'])

rutasAxis = np.arange(len(mostCommonTrips))
fig = plt.figure()
plt.figure(figsize=(12,6))
fig.suptitle('Rutas mas frecuentes', fontsize = 18)
plt.ylabel('Cantidad de Viajes', fontsize = 12)
my_xticks = rutas
plt.xticks(rutasAxis, my_xticks, fontsize = 10, rotation='vertical')
axes = plt.gca()

plt.bar(rutasAxis, cantViajes, align='center', color='brown')




'''
# Grafico en el mapa de las rutas mas frecuentes
myGmap4 = gmaps.Map()

stationMarkers = []
for index, row in mostCommonTrips.iterrows():
    startStation = stations.loc[stations['name'] == str(row['start_station_name'])]
    endStation = stations.loc[stations['name'] == str(row['end_station_name'])]
    
    # Estaciones
    stationMarkers.append({"name": row['start_station_name'], "location": (float(startStation['lat']), float(startStation['long']))})
    stationMarkers.append({"name": row['end_station_name'], "location": (float(endStation['lat']), float(endStation['long']))})
    
    # Caminos
    station1 = (float(startStation['lat']), float(startStation['long']))
    station2 = (float(endStation['lat']), float(endStation['long']))
    caminoI = gmaps.directions_layer(station1, station2)
    myGmap4.add_layer(caminoI)


# Remove duplicated stations    
stationMarkersSet = []
for x in stationMarkers:
    if x not in stationMarkersSet:
        stationMarkersSet.append(x)

    
stationLocations = [station["location"] for station in stationMarkersSet]
infoBoxTemplate = """
<dl>
<dt>Name</dt><dd>{name}</dd>
</dl>
"""
stationInfo = [infoBoxTemplate.format(**station) for station in stationMarkersSet]
stations_layer = gmaps.marker_layer(stationLocations, info_box_content=stationInfo)
myGmap4.add_layer(stations_layer)

myGmap4 
'''




stations.groupby('city').count()
matriz = [[0 for x in range(len(stations.id))] for y in range(len(stations.id))]
matriz = pd.DataFrame(matriz)
matriz.rename(index=stations.id, columns=stations.id, inplace=True)
matriz




df2 = pd.DataFrame(trips.loc[:,['start_station_id','end_station_id']].groupby('start_station_id').end_station_id.value_counts())
df2.rename(columns={'end_station_id':'count'}, inplace=True)
df2.reset_index(inplace=True)
df2




for row in df2.values:
    matriz.loc[matriz.index == row[0],row[1]] = row[2]
matriz




fig2, ax2 = plt.subplots(1,1)
fig2.set_size_inches((14,12))

color = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, as_cmap=True)
sns.heatmap(matriz, cmap=color, ax=ax2)
ax2.invert_yaxis()
ax2.set_xticklabels(stations.name, rotation='vertical')
ax2.set_yticklabels(stations.name.sort_index(ascending=False))
("")









viajesMasDe30 = trips[trips['duration'] > 18000]
len(viajesMasDe30) / len(trips)
viajesMasDe30['subscription_type'].value_counts()




customersMas30 = viajesMasDe30[viajesMasDe30['subscription_type'] == 'Customer']

customersMenos30 =  trips[trips['duration'] < 18000]
customersMenos30 = customersMenos30[customersMenos30['subscription_type'] == 'Customer']
len(customersMenos30)




len(customersMas30[customersMas30['duration'] < 36000])




viajesMenosDe60 = viajesMasDe30[viajesMasDe30['duration'] < 36000] 
print('Cantidad de viajes de mas de 30 minutos', len(viajesMasDe30))
print('\nCantidad de viajes de mas de 30 minutos y menos de 60 minutos', len(viajesMenosDe60))
print('\nCantidad de viajes de menos de 30 minutos', len(trips)- len(viajesMasDe30) )




days = viajesMenosDe60.loc[:,'start_date']
days = pd.to_datetime(days, format='%m/%d/%Y %H:%M')
days.head()




days.dt.date.value_counts().plot()




days.dt.date.value_counts().mean()




days.dt.year.value_counts()




# Comentado para Kaggle porque usamos un set de datos externo. Ver en Github.
'''
zips = pd.read_csv('../input/free-zipcode-database-Primary.csv', index_col=0, low_memory=False).reset_index()
ciudadesEstados = zips[zips.columns[2:4]].set_index('City')

Se analizan los zip codes de cada usuario, se supone que los zip codes son el zip code del domicilio de cada usuario. Se limpia el csv de trips con los viajes que tienen zip codes validos, sea por que el campo es nil, NaN o tiene un formato no valido.

### Generado CSV con los viajes con zip code valido

zips = pd.read_csv('../input/free-zipcode-database-Primary.csv', index_col=0, low_memory=False)

def findStateByZip(zipcode):
    try:
        city = cityState.loc[int(zipcode)][0]
        state = cityState.loc[int(zipcode)][1]
    except:
        state = 'nan'
        city = 'nan'
    return (city, state)
    
## Importo el dataset de subscribers con zips validos    
validCities = pd.read_csv('../input/valid_zip_codes_subscribers.csv')

## Grafico de ciudades uso de bicicletas por la cuidad de residencia, sacando San Francisco que es bastante

topCiudades = validCities['user_city'].value_counts()
topCiudades.head()  / topCiudades.sum()

topCiudades.head(20).plot.bar()

Las ciudades que tienen estaciones de bicicletas son: San jose, RedWood City, San Francisco, Palo alto, mountain view, si las saco del top, puedo ver que viajes son propios de usuarios que no viven en las ciudades que tienen estaciones.

topCiudadesSinLocales = topCiudades.drop(['SAN JOSE', 'SAN FRANCISCO', 'REDWOOD CITY', 'PALO ALTO', 'MOUNTAIN VIEW'])
(topCiudadesSinLocales / topCiudadesSinLocales.sum()).head(20).sum()

topCiudadesSinLocales.head(20).plot.bar()

### Grafico de cantidad de viajes segun estado del usuario

total = len(validCities)
topEstados = validCities['user_state'].value_counts()
topEstados.map(lambda x: ' total %3.3f%% ' % (x * 100/total)).head(20)

Si se deja afuera a california se puede analizar mejor los otros estados, ya que california es el 99% de los estados de los usuarios subscriptos

topEstados.drop('CA').head(15).plot.bar()

## Universidades

nameCityStations = stations[[1,5]].set_index('name')
nameCityStations.loc['Stanford in Redwood City']
stations[stations['name'] == 'Stanford in Redwood City']

validCities['end_station_name'].value_counts().head()

stanfordLocation = [37.427529, -122.170354]

def distance(start, endx, endy):
    value = abs(start[0]-endx) + abs(start[1]-endy) #distancia Manhattan
    return value

stationsLongLat = stations[[2,3]]

longitudesCercaStanford = stationsLongLat.apply(distance, axis=1, args=stanfordLocation).sort_values()
longitudesCercaStanford.head()

estacionesCercaDeStanford = stations.loc[longitudesCercaStanford.index]
estacionesCercaDeStanford.head(5)

topEndStations = validCities['end_station_name'].value_counts()
topEndStations = topEndStations.reset_index().rename(columns={'index':'name', 'end_station_name':'number_of_trips'})
topEndStations.head()

topEndStations
estacionesCercaConPosicionTop = pd.merge(estacionesCercaDeStanford, topEndStations.reset_index(), on='name')[[1,7,8]].rename(columns={'index':'posicion_top_viajes'})
estacionesCercaConPosicionTop.head()

def estacionesCercaDeLocation(location):
    stationsLongLat = stations[[2,3]]
    longitudesCerca = stationsLongLat.apply(distance, axis=1, args=location).sort_values()
    #print(longitudesCerca.head())
    estacionesCerca = stations.loc[longitudesCerca.index]
    topEndStations = validCities['end_station_name'].value_counts()
    topEndStations = topEndStations.reset_index().rename(columns={'index':'name', 'end_station_name':'number_of_trips'})
    estacionesCercaConPosicionTop = pd.merge(estacionesCerca, topEndStations.reset_index(), on='name')[[1,7,8]].rename(columns={'index':'posicion_top_viajes'})
    return estacionesCercaConPosicionTop
    
estacionesCercaDeLocation(stanfordLocation).head(3)

uccfLocation = [37.725700, -122.451583]
estacionesCercaDeLocation(uccfLocation).head(3)

ucBerkley = [37.872378, -122.258481]
estacionesCercaDeLocation(ucBerkley).head(3)

# Viajes Sin SF ni SJ

viajesSinSFSJ = validCities[(validCities['user_city'] != 'SAN FRANCISCO') & (validCities['user_city'] != 'SAN JOSE')]
viajesSinSFSJ['start_station_name'].value_counts().head(8)

viajesSinSFSJ['end_station_name'].value_counts().head()

## Viajes de Berkeley y Oakland

viajesBROA = validCities[(validCities['user_city'] == 'BERKELEY') | (validCities['user_city'] == 'OAKLAND')]
viajesBROA['start_station_name'].value_counts().head()

Ciudades que mas usan el transbay terminal

validCities[validCities['start_station_name'] == 'Embarcadero at Sansome']['user_city'].value_counts().head()

validCities[validCities['user_city'] == 'STANFORD'].head()

validCities[validCities['start_station_name'] == 'San Francisco Caltrain (Townsend at 4th)']['user_city'].value_counts().head(8)

validCities[validCities['start_station_name'] == 'Temporary Transbay Terminal (Howard at Beale)']['user_city'].value_counts().head(8)

validCities[validCities['user_city'] == 'RICHMOND']['end_station_name'].value_counts().head()

validCities[validCities['user_city'] == 'SUNNYVALE']['end_station_name'].value_counts().head()

validCities[validCities['user_city'] == 'SAN RAFAEL']['end_station_name'].value_counts().head()

viajesSinSanFrancisco = validCities[(validCities['user_city'] != 'SAN FRANCISCO')]

viajesSinSanFrancisco['start_station_name'].value_counts().head(20).plot.bar()

Ciudades que se toman el caltrain

viajesDeCaltrainInicio = viajesSinSanFrancisco[(viajesSinSanFrancisco['start_station_name'] == 'San Francisco Caltrain (Townsend at 4th)') | (viajesSinSanFrancisco['start_station_name'] == 'San Francisco Caltrain 2 (330 Townsend)')]
viajesDeCaltrainFin = viajesSinSanFrancisco[(viajesSinSanFrancisco['end_station_name'] == 'San Francisco Caltrain (Townsend at 4th)') | (viajesSinSanFrancisco['end_station_name'] == 'San Francisco Caltrain 2 (330 Townsend)')]

ciudadesQueSeTomanElCaltrainInicio = viajesDeCaltrainInicio['user_city'].value_counts()
ciudadesQueSeTomanElCaltrainFin = viajesDeCaltrainFin['user_city'].value_counts()

ciudadesQueSeTomanElCaltrainInicio.head()
#ciudadesQueSeTomanElCaltrainFin.head()


viajesDeMetroInicio = viajesSinSanFrancisco[(viajesSinSanFrancisco['start_station_name'] == 'Beale at Market') | (viajesSinSanFrancisco['start_station_name'] == 'Market at Sansome')]
viajesDeMetroInicio['user_city'].value_counts().head()

viajesDeColectivoInicio = viajesSinSanFrancisco[(viajesSinSanFrancisco['start_station_name'] == 'Temporary Transbay Terminal (Howard at Beale)')]
viajesDeColectivoInicio['user_city'].value_counts().head()

viajesDeColectivoInicio = viajesSinSanFrancisco[(viajesSinSanFrancisco['start_station_name'] == 'Harry Bridges Plaza (Ferry Building)')]
viajesDeColectivoInicio['user_city'].value_counts().head()


Las estaciones mas frecuentadas por la gente que no vive en san francisco son las que estan cerca del metro, como Beale at Market o Market at Sansome, que conectan las zonas entre san francisco y las ciudades de Oakland, Richmond, Berkeley o Alameda. Las estaciones que estan cerca del Caltrain, que conectan toda la zona desde San Francisco hasta San Jose, son dos que estan en Townsend at 4th. Finalmente la Temporary Transbay Terminal en Howard at Beale, es un destino de colectivos de toda la region que se conoce como East Bay Region. En Harry Bridges Plaza tambien hay un destino de colectivos, pero principalmente del norte de san francisco, ciudades como San Rafale o Vallejo

caltrain = validCities[(validCities['start_station_name'] == 'San Francisco Caltrain (Townsend at 4th)') | (validCities['start_station_name'] == 'San Francisco Caltrain 2 (330 Townsend)')]

def rankingCities(nameStation):
    print(validCities[validCities['start_station_name'] == nameStation]['user_city'].value_counts().head(8))
    
rankingCities('San Francisco Caltrain (Townsend at 4th)')

'''


























































































































































































