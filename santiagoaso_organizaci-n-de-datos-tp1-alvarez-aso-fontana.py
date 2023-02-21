#!/usr/bin/env python
# coding: utf-8








# carga del Dataframe Trip
trip = pd.read_csv('../input/trip.csv', low_memory=False)
weather=pd.read_csv('../input/weather.csv', low_memory=False)
station=pd.read_csv('../input/station.csv', low_memory=False)
trip.info()

















































# carga de un data frame
weather = pd.read_csv('../input/weather.csv', low_memory=False)




# vemos primeras filas del data frame
weather[:3]




# si queremos analizar cuales son los valores de las columnas podemos obtenerlos con .columns
print ("COLUMNS WEATHER.CSV:")
print ("")
for name in weather.columns.values:
    print (name)
    # si queremos analizar cuales son los valores de las columnas podemos obtenerlos con .columns
print ("")




weatherSmall = weather.loc[:,("date","mean_temperature_f","mean_dew_point_f","mean_humidity","mean_sea_level_pressure_inches","mean_visibility_miles","mean_wind_speed_mph","precipitation_inches","cloud_cover","events", "wind_dir_degrees")]
weatherSmall.head()




weatherSmall.describe()




weatherGrouped = weatherSmall.groupby("events").mean()
weatherGrouped.head()




weatherEvents = weather.loc[(weather.events > ""),("date","events")]
weatherEvents.head()




#Analizamos los datos Stations
stations = pd.read_csv('../input/station.csv', low_memory=False)




stations[:10]




#Vemos cuantas hay por ciudad
count_citys = stations['city'].value_counts()
count_citys




get_ipython().run_line_magic('matplotlib', 'notebook')
count_citys[:10].plot('bar')




#Vemos cuantos hay por cantidad de "dock"
count_dock = stations['dock_count'].value_counts()
count_dock




#Vemos fecha de instalacion.
# Se remarca que se instalo mas al inico que al final. Suponemos que las obras estarian "terminadas" ya
count_isntalation = stations['installation_date'].value_counts()
count_isntalation 




#los demas datos y los de clima trabajan por zipCode. Por lo que "traducimos" los zipCode
tabla = {
        'zipCode': ['95113', '94063', '94041', '94107', '94301'],
        'city': ['San Jose', 'Redwood City', 'Mountain View', 'San Francisco', 'Palo Alto'],
}

tabla
df_a = pd.DataFrame(tabla, columns = ['zipCode', 'city'])
df_a




Tara = pd.merge(stations, df_a, on='city', how='right')
Tara[:10]




#Reducimos los datos a las Id y Los ZipCode.
tablaIdZip = Tara.loc[:,("id","zipCode")]




# carga de un data frame
#flights = pd.read_csv('../data/flight-delays/flights.csv', low_memory=False)
status = pd.read_csv('../input/status.csv', low_memory=True, parse_dates=['time'])
status.time = status.time.dt.date
status.head()




status[:10]




#vemos la cantidad de reportes por cantidad de bicis
NotBike = status['bikes_available'].value_counts()
NotBike[:10]




#Lo mas importante es saber cuando No Hay bicis. En esos casos no estamos cumpliendo con el servicio ni con los clientes.
NotBike = status.loc[:,['station_id','bikes_available','time']].groupby('bikes_available')
NotBike = NotBike.get_group(0)
NotBike[:10]




#Estaciones con mas reportes de falta de bisicleta.
NotBike1 = NotBike['station_id'].value_counts()
NotBike1[:10]




#Lo emparejamos con el Codigo Zip. para poder comprarlo con el Clima
NotBikeZip = pd.merge(NotBike, tabla, on='station_id', how='right')
NotBikeZip[:10]




#reducimos a lo util
NotBike = NotBikeZip.loc[:,("time","id","zipCode")]
NotBike[:10]




#guradmos
NotBike.to_csv("NotBike.csv")




Revisamos lo mismo con los Docks
NotDocks = status['docks_available'].value_counts()
NotDocks[:10]




# Otra ves el caso intersante es que halla 0 Docks. Es implica que uno llega con la bici y no la puede guadar
# Un fallo en el servicio
NotDocks = status.loc[:,['station_id','docks_available','time']].groupby('docks_available')
NotDocks = NotDocks.get_group(0)
NotDocks[:10]




#Los lugares con mallor falta de Dock. No podemos estimar cuantos faltan agregar
NotDocks1 = NotDocks['station_id'].value_counts()
NotDocks1[:10]




NotDocksZip = pd.merge(NotDocks, tabla, on='station_id', how='right')
NotDocksZip[:10]




NotDocks = NotBikeZip.loc[:,("time","id","zipCode")]
NotDocks[:10]




#buscamos si hay una relacion entre la cantidad de Docks libre y bisicletas
grouped  = status.loc[:,['station_id','bikes_available','docks_available']].groupby('station_id')
Station = grouped.mean()
Station[:10]




#Resulta que si.
#Cosa que nos lleva a pensar que no hay una relacion con la hubicacion del lugar con que halla o no bisicletas
#Por lo que se puede estimar que es algo mas relativo a la hora. 
Station.loc[:,['bikes_available','docks_available']].corr()

