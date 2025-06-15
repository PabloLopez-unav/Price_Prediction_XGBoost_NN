import pandas as pd
import matplotlib.pyplot as plt
import squarify  # para treemap

# Datos
data = {
    'Feature': [
        'CONSTRUCTEDAREA', 'Zona_Cuadricula', 'BUILTTYPEID_3', 'HASLIFT', 'DISTANCE_TO_CASTELLANA',
        'BATHNUMBER', 'SWIMMINGPOOL', 'lat_normalized', 'DIST_CITY_CENTER', 'BUILTTYPEID_1',
        'lon_normalized', 'AC AA', 'BUILTTYPEID_2', 'DOORMAN', 'PARKING SPACE',
        'FLOORCLEAN', 'ROOM NUM', 'REST of 14 VARIABLES (7.4%)'
    ],
    'Importance': [
        0.288333, 0.082039, 0.072755, 0.070262, 0.065788, 0.064307, 0.050757, 0.045587, 0.032205,
        0.029080, 0.027149, 0.019754, 0.018715, 0.015273, 0.014078, 0.012388, 0.009758, 0.074447
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
    text_kwargs={'fontsize': 19},
    color=['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#9999FF', '#CC99FF', '#FF99CC'] * 4
    + ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFFF', '#9999FF', '#CC99FF', '#FF99CC'][:len(df) - 36]
)
plt.axis('off')
plt.title('Importancia de las Features según XGBoost', fontsize=30)
plt.tight_layout()
plt.savefig('8.0 Graficas varias/TreeMap_XGBoost_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.show()
