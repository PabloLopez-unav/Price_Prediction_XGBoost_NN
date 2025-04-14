import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_grid_zone(input_file, output_file, grid_size=50):
    # Leer el archivo CSV original
    df = pd.read_csv(input_file)

    # Definir los límites de las coordenadas
    lon_min, lon_max = -3.768, -3.608
    lat_min, lat_max = 40.346, 40.498

    # Normalizar longitud y latitud a índices de la cuadrícula (0-99)
    df['lon_normalized'] = ((df['LONGITUDE'] - lon_min) / (lon_max - lon_min)) * (grid_size - 1)
    df['lat_normalized'] = ((df['LATITUDE'] - lat_min) / (lat_max - lat_min)) * (grid_size - 1)

    # Convertir las coordenadas normalizadas a números de zona (1-10000)
    df['Zona_Cuadricula'] = (df['lat_normalized'].astype(int) * grid_size + df['lon_normalized'].astype(int)) + 1

    # Guardar el resultado en un nuevo archivo CSV
    df.to_csv(output_file, index=False)
    print(f"Archivo con las zonas generado: {output_file}")
    
    return df

def create_heatmaps(df, grid_size, save_path=None):
    """
    Crea heatmaps usando el sistema de Zona_Cuadricula exacto que definiste
    
    Args:
        df: DataFrame con la columna Zona_Cuadricula
        grid_size: Tamaño de la cuadrícula (50 para una 50x50)
        save_path: Opcional, ruta para guardar la imagen
    """
    # Crear matriz vacía para los heatmaps
    density_matrix = np.zeros((grid_size, grid_size))
    price_matrix = np.zeros((grid_size, grid_size))
    
    # Calcular valores para cada celda
    for zone in range(1, grid_size**2 + 1):
        # Obtener coordenadas (x,y) de la zona
        y = (zone - 1) // grid_size
        x = (zone - 1) % grid_size
        
        # Filtrar propiedades en esta zona
        zone_data = df[df['Zona_Cuadricula'] == zone]
        
        # Calcular métricas
        density_matrix[y, x] = len(zone_data)
        price_matrix[y, x] = zone_data['PRICE'].mean() if not zone_data.empty else 0
    
    # Configurar los plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap de densidad
    sns.heatmap(density_matrix, cmap='viridis', ax=ax1, 
                cbar_kws={'label': 'Número de viviendas'})
    ax1.set_title(f'Densidad de Viviendas (Total: {len(df)} propiedades)')
    ax1.set_xlabel('Coordenada X')
    ax1.set_ylabel('Coordenada Y')
    ax1.invert_yaxis()  # Para que Y=0 esté abajo
    
    # Heatmap de precios
    sns.heatmap(price_matrix, cmap='plasma', ax=ax2,
                cbar_kws={'label': 'Precio medio (€)'})
    ax2.set_title('Precio Medio por Zona')
    ax2.set_xlabel('Coordenada X')
    ax2.set_ylabel('Coordenada Y')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
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

    grid_size = 50 

    # 1. Generar las zonas de cuadrícula
    df_with_grid = generate_grid_zone(input_file, output_file, grid_size)
    
    # 2. Crear y mostrar los heatmaps
    create_heatmaps(df_with_grid, grid_size, save_path=heatmaps_path)
        