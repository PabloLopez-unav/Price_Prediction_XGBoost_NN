import pandas as pd
import numpy as np

def generate_grid_zone(input_file, output_file):
    # Leer el archivo CSV original
    df = pd.read_csv(input_file)

    # Definir los límites de las coordenadas
    lon_min, lon_max = -3.768, -3.608
    lat_min, lat_max = 40.346, 40.498

    # Calcular las dimensiones de la cuadrícula (100x100)
    grid_size = 100

    # Normalizar longitud y latitud a índices de la cuadrícula (0-99)
    df['lon_normalized'] = ((df['LONGITUDE'] - lon_min) / (lon_max - lon_min)) * (grid_size - 1)
    df['lat_normalized'] = ((df['LATITUDE'] - lat_min) / (lat_max - lat_min)) * (grid_size - 1)

    # Convertir las coordenadas normalizadas a números de zona (1-10000)
    df['Zona_Cuadricula'] = (df['lat_normalized'].astype(int) * grid_size + df['lon_normalized'].astype(int)) + 1

    # Guardar el resultado en un nuevo archivo CSV con solo las columnas necesarias
    df[['Zona_Cuadricula']].to_csv(output_file, index=False)

    print(f"Archivo con las zonas generado: {output_file}")

# Llamada a la función con el archivo de entrada y salida
input_file = "C:/Users/costa/Desktop/TFG/4.0 Estadística Básica para entender el data set/4.2 Funciones y cosas varias/Madrid_Sale.csv"
output_file = "C:/Users/costa/Desktop/TFG/4.0 Estadística Básica para entender el data set/4.2 Funciones y cosas varias/zonas_cuadricula.csv"
generate_grid_zone(input_file, output_file)
