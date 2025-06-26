import pandas as pd
import matplotlib.pyplot as plt
import squarify  # para treemap

# Datos
data = {
    'Feature': [
        'CONSTRUCTEDAREA', 'Zona_Cuadricula', 'HASLIFT', 'BATHNUMBER',
        'DIST_CASTELLANA', 'SWIMMINGPOOL', 'lat_normalized',
        'DIST_CITY_CENTER', 'lon_normalized', 'BUILTTYPEID_1',
        'BUILTTYPEID_2', 'AC/AA', 'PARKINGSPACE',
        'FLOORCLEAN', 'DOORMAN', 'WARDROBE', 'DUPLEX',
        'FLOORS', 'FLAT LOC', 'ROOMS',
        'REST of 16 VARIABLES (6.69%)'      # suma de las 14 importancias más bajas
    ],
    'Importance': [
        0.307139, 0.087857, 0.080460, 0.076665, 0.073946,
        0.047019, 0.037968, 0.034546, 0.029475, 0.025765,
        0.023280, 0.021241, 0.018486, 0.014430, 0.012524,
        0.008997, 0.008944, 0.008277, 0.008180, 0.007895,
        0.066905
    ]
}

df = pd.DataFrame(data)

# Ordenar de mayor a menor
df = df.sort_values('Importance', ascending=False)

# Crear el treemap
plt.figure(figsize=(17, 10))
squarify.plot(
    sizes=df['Importance'],
    label=df['Feature'],
    alpha=0.8,
    text_kwargs={'fontsize': 16.5},
    color=['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#9999FF', '#CC99FF', '#FF99CC'] * 4
    + ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#9999FF', '#CC99FF', '#FF99CC'][:len(df) - 36]
)
plt.axis('off')
plt.title('Importancia de las Features según XGBoost', fontsize=30)
plt.tight_layout()
plt.savefig('8.0 Graficas varias/TreeMap_XGBoost_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.show()
