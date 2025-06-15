import pandas as pd
import matplotlib.pyplot as plt
import squarify
import os

# Crear carpeta si no existe
os.makedirs("8.0 Graficas varias", exist_ok=True)

# Datos
data_nn = {
    'Feature': [
        'CONSTRUCTEDAREA', 'CADDWELLINGCOUNT', 'DIST_TO_CASTELLANA', 'FLOORCLEAN',
        'CADCONSTRUCTIONYEAR', 'BATHNUMBER', 'ROOMNUMBER', 'Zona_Cuadricula',
        'CADMAXBUILDFLOOR', 'DIST_TO_CITY_CENTER', 'DIST_TO_METRO', 'WARDROBE',
        'PARKINGSPACE', 'BOXROOM', 'TERRACE', 'DOORMAN', 'AIRCONDITIONING',
        'SWIMMINGPOOL', 'LIFT', 'GARDEN', 'ISDUPLEX', 'ISINTOPFLOOR', 'ISSTUDIO'
    ],
    'Importance': [
        0.063851, 0.057398, 0.054933, 0.054794, 0.052064, 0.051975, 0.051079, 0.050983,
        0.049350, 0.047047, 0.046295, 0.042637, 0.039929, 0.039637, 0.039339, 0.038835,
        0.038739, 0.038512, 0.038361, 0.036894, 0.023937, 0.022944, 0.020466
    ]
}

df_nn = pd.DataFrame(data_nn)
df_nn = df_nn.sort_values('Importance', ascending=False)

# Crear gráfico
plt.figure(figsize=(17, 10))
squarify.plot(
    sizes=df_nn['Importance'],
    label=df_nn['Feature'],
    alpha=0.8,
    text_kwargs={'fontsize': 19},
    color=['#FF9999', '#CCFF99', '#FFFF99', '#FFCC99', '#99FF99', '#99FFFF', '#9999FF', '#CC99FF', '#FF99CC'] * 3
)
plt.axis('off')
plt.title('Importancia de las Features según Red Neuronal (Primera capa)', fontsize=30)
plt.tight_layout()
plt.savefig('8.0 Graficas varias/TreeMap_NN_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.show()
