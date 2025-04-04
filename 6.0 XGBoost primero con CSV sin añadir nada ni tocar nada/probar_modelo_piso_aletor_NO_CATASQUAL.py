import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib


# Cargar el modelo guardado
modelo_cargado = joblib.load("modelo_xgboost_60per_NO_CATASQUAL.pkl")

# Crear un DataFrame con los datos de un solo piso (asegúrate de que tenga las mismas columnas que el modelo)
piso_nuevo = pd.DataFrame({
    "CONSTRUCTEDAREA": [83],
    "ROOMNUMBER": [3],
    "BATHNUMBER": [1],
    "HASTERRACE": [1],
    "HASLIFT": [0],
    "HASAIRCONDITIONING": [0],
    "AMENITYID": [2],  # Ajustar según los valores posibles
    "HASPARKINGSPACE": [0],
    "ISPARKINGSPACEINCLUDEDINPRICE": [0],
    "PARKINGSPACEPRICE": [0],
    "HASNORTHORIENTATION": [0],
    "HASSOUTHORIENTATION": [0   ],
    "HASEASTORIENTATION": [0],
    "HASWESTORIENTATION": [1],
    "HASBOXROOM": [0],
    "HASWARDROBE": [0],
    "HASSWIMMINGPOOL": [0],
    "HASDOORMAN": [0],
    "HASGARDEN": [0],
    "ISDUPLEX": [0],
    "ISSTUDIO": [0],
    "ISINTOPFLOOR": [0],
    "FLOORCLEAN": [1],  # Suponiendo que es el segundo piso
    "FLATLOCATIONID": [1],  # Ajustar según el dataset
    "CADCONSTRUCTIONYEAR": [1900],
    "CADMAXBUILDINGFLOOR": [2],
    "CADDWELLINGCOUNT": [5],
    "BUILTTYPEID_1": [0],
    "BUILTTYPEID_2": [0],
    "BUILTTYPEID_3": [1],
    "DISTANCE_TO_CITY_CENTER": [3.7],
    "DISTANCE_TO_METRO": [0.3],
    "DISTANCE_TO_CASTELLANA": [1.4],  # Distancia a una calle principal
    "LONGITUDE": [-3.672122],  # Coordenadas de Madrid como ejemplo
    "LATITUDE": [40.436377]

    
})



# Hacer la predicción
precio_predicho = modelo_cargado.predict(piso_nuevo)

print(f"El precio predicho para el piso es: {precio_predicho[0]:,.2f} euros")


precio_con_inflacion = precio_predicho * 1.44  # Inflación calculada con datos de Idealista 
# https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/venta/madrid-comunidad/madrid-provincia/madrid/historico/


print(f"El precio predicho para el piso es: {precio_con_inflacion[0]:,.2f} euros")