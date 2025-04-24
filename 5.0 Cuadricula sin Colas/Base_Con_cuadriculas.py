import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_grid_zone(input_file, output_file, grid_size=60):
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
    Crea heatmaps separados para densidad, precio medio y precio por m2
    
    Args:
        df: DataFrame con la columna Zona_Cuadricula
        grid_size: Tamaño de la cuadrícula (50 para una 50x50)
        save_path: Opcional, ruta base para guardar las imágenes (sin extensión)
    """
    # Crear matrices vacías para los heatmaps
    density_matrix = np.zeros((grid_size, grid_size))
    price_matrix = np.zeros((grid_size, grid_size))
    price_m2_matrix = np.zeros((grid_size, grid_size))
    
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
        price_m2_matrix[y, x] = zone_data['UNITPRICE'].mean() if not zone_data.empty else 0
    
    #Precio medio m2 total
    price_m2_total = df['UNITPRICE'].mean()
    print(f"Precio medio por m2 total: {price_m2_total:.2f} €")
    
    # Heatmap de densidad
    plt.figure(figsize=(10, 8))
    sns.heatmap(density_matrix, cmap='viridis', 
                cbar_kws={'label': 'Número de viviendas'})
    plt.title(f'Densidad de Viviendas (Total: {len(df)} propiedades)')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.gca().invert_yaxis()  # Para que Y=0 esté abajo
    
    if save_path:
        density_path = f"{save_path}_densidad.png"
        plt.savefig(density_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap de densidad guardado en: {density_path}")
    
    plt.show()
    
    # Heatmap de precios medios
    plt.figure(figsize=(10, 8))
    sns.heatmap(price_matrix, cmap='plasma',
                cbar_kws={'label': 'Precio medio (€)'})
    plt.title('Precio Medio por Zona')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.gca().invert_yaxis()
    
    if save_path:
        price_path = f"{save_path}_precio_medio.png"
        plt.savefig(price_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap de precio medio guardado en: {price_path}")
    
    plt.show()
    
    # Heatmap de precios por m2
    plt.figure(figsize=(10, 8))
    sns.heatmap(price_m2_matrix, cmap='magma',
                cbar_kws={'label': 'Precio medio por m² (€)'})
    plt.title('Precio Medio por Metro Cuadrado')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.gca().invert_yaxis()
    
    if save_path:
        price_m2_path = f"{save_path}_precio_m2.png"
        plt.savefig(price_m2_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap de precio por m2 guardado en: {price_m2_path}")
    
    plt.show()

# Proceso principal
if __name__ == "__main__":
    # Configuración de rutas
    input_file = r"C:\Users\costa\Desktop\TFG\5.0 Cuadricula sin Colas\filtered_dataset_NO_TAILS.csv"
    output_file = r"C:\Users\costa\Desktop\TFG\5.0 Cuadricula sin Colas\Dataset_Con_Filtrado_y_Cuadrículas.csv"
    heatmaps_base_path = r"C:\Users\costa\Desktop\TFG\5.0 Cuadricula sin Colas\heatmap"

    grid_size = 60 

    # 1. Generar las zonas de cuadrícula
    df_with_grid = generate_grid_zone(input_file, output_file, grid_size)
    
    # 2. Crear y mostrar los heatmaps
    create_heatmaps(df_with_grid, grid_size, save_path=heatmaps_base_path)