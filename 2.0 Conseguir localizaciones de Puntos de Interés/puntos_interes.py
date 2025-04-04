import overpy
import pandas as pd

# Inicializar la API de Overpass
api = overpy.Overpass()

# Consulta para obtener hospitales y centros de salud en Madrid
query = """
[out:json];
area[name="Madrid"]->.searchArea;
(
  node["amenity"="hospital"](area.searchArea);
  node["amenity"="clinic"](area.searchArea);    
  node["healthcare"="hospital"](area.searchArea);
  node["healthcare"="clinic"](area.searchArea);
);
out body;
"""

# Ejecutar la consulta
result = api.query(query)

# Extraer datos en una lista
data = []
for node in result.nodes:
    name = node.tags.get("name", "Desconocido")
    lat = node.lat
    lon = node.lon
    data.append([name, lat, lon])

# Convertir a DataFrame
df = pd.DataFrame(data, columns=["Nombre", "Latitud", "Longitud"])

# Guardar como CSV
df.to_csv("hospitales_madrid_3.csv", index=False, encoding="utf-8-sig")


print("Datos guardados en hospitales_madrid.csv")
