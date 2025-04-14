import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Guardar el resultado en un nuevo archivo CSV
    df.to_csv(output_file, index=False)
    print(f"Archivo con las zonas generado: {output_file}")
    
    return df

def create_heatmaps(df, save_path=None):
    """
    Crea y muestra dos heatmaps:
    1. Densidad de viviendas por zona
    2. Precio medio por zona
    
    Parameters:
        df (DataFrame): DataFrame con los datos
        save_path (str): Ruta para guardar la imagen (opcional)
    """
    # Preparar datos para los heatmaps
    grid_counts = df.groupby(['lat_normalized', 'lon_normalized']).size().reset_index(name='counts')
    grid_prices = df.groupby(['lat_normalized', 'lon_normalized'])['PRICE'].mean().reset_index(name='avg_price')

    # Crear matrices para los heatmaps
    count_matrix = grid_counts.pivot(index='lat_normalized', 
                                   columns='lon_normalized', 
                                   values='counts').fillna(0)
    
    price_matrix = grid_prices.pivot(index='lat_normalized', 
                                   columns='lon_normalized', 
                                   values='avg_price').fillna(0)

    # Configurar los subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap 1: Número de viviendas por zona
    sns.heatmap(count_matrix, cmap='viridis', ax=ax1)
    ax1.set_title('Densidad de Viviendas por Zona')
    ax1.set_xlabel('Longitud (normalizada)')
    ax1.set_ylabel('Latitud (normalizada)')

    # Heatmap 2: Precio medio por zona
    sns.heatmap(price_matrix, cmap='plasma', ax=ax2)
    ax2.set_title('Precio Medio por Zona (€)')
    ax2.set_xlabel('Longitud (normalizada)')
    ax2.set_ylabel('Latitud (normalizada)')

    plt.tight_layout()
    
    # Guardar la imagen si se especifica una ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmaps guardados en: {save_path}")
    
    plt.show()

# Proceso principal
if __name__ == "__main__":
    # Configuración de rutas
    input_file = r"C:\Users\costa\Desktop\TFG\5.0 Cuadricula sin Colas\filtered_dataset_NO_TAILS.csv"
    output_file = r"C:\Users\costa\Desktop\TFG\5.0 Cuadricula sin Colas\Dataset_Con_Filtrado_y_Cuadrículas.csv"
    heatmaps_path = r"C:\Users\costa\Desktop\TFG\5.0 Cuadricula sin Colas\heatmaps.png"

    # 1. Generar las zonas de cuadrícula
    df_with_grid = generate_grid_zone(input_file, output_file)
    
    # 2. Crear y mostrar los heatmaps
    create_heatmaps(df_with_grid, save_path=heatmaps_path)