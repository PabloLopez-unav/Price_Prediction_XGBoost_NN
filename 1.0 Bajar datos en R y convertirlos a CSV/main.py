import os
import rdata
import urllib
import pandas as pd  # Aseg�rate de importar pandas

file_names = ["Barcelona_POIS.rda", "Barcelona_Polygons.rda", "Barcelona_Sale.rda", "Madrid_POIS.rda",
              "Madrid_Polygons.rda",  "Madrid_Sale.rda", "Valencia_POIS.rda", "Valencia_Polygons.rda",
              "Valencia_Sale.rda", "properties_by_district.rda"]


def download_rdata():
    """
    Function that downloads the .rda files from Idealista repository.
    For each file, it is verified beforehand if the file is already in the /data folder in order to
    avoid unnecessary downloads
    """
    for rda_file in file_names:
        file_path = "data/" + rda_file
        if not os.path.isfile(file_path):
            urllib.request.urlretrieve("https://github.com/paezha/idealista18/raw/master/data/" + rda_file, file_path)
        else:
            continue


def read_data():
    """
    For each file located in /data folder, extract the information as a dataframe or a dictionary containing data
    During the execution of this function UserWarning related with rdata package may appear but can be ignored
    """
    datasets = {}

    barcelona_pois = rdata.read_rda("data/Barcelona_POIS.rda")
    barcelona_polygons = rdata.read_rda("data/Barcelona_Polygons.rda", default_encoding="utf8")
    barcelona_sale = rdata.read_rda("data/Barcelona_Sale.rda")
    madrid_pois = rdata.read_rda("data/Madrid_POIS.rda")
    madrid_polygons = rdata.read_rda("data/Madrid_Polygons.rda", default_encoding="utf8")
    madrid_sale = rdata.read_rda("data/Madrid_Sale.rda")
    valencia_pois = rdata.read_rda("data/Valencia_POIS.rda")
    valencia_polygons = rdata.read_rda("data/Valencia_Polygons.rda", default_encoding="utf8")
    valencia_sale = rdata.read_rda("data/Valencia_Sale.rda")
    properties_by_district = rdata.read_rda("data/properties_by_district.rda")

    datasets['Barcelona_POIS'] = barcelona_pois["Barcelona_POIS"]
    datasets['Barcelona_Polygons'] = barcelona_polygons['Barcelona_Polygons']
    datasets['Barcelona_Sale'] = barcelona_sale['Barcelona_Sale']
    datasets['Madrid_POIS'] = madrid_pois['Madrid_POIS']
    datasets['Madrid_Polygons'] = madrid_polygons['Madrid_Polygons']
    datasets['Madrid_Sale'] = madrid_sale['Madrid_Sale']
    datasets['Valencia_POIS'] = valencia_pois['Valencia_POIS']
    datasets['Valencia_Polygons'] = valencia_polygons['Valencia_Polygons']
    datasets['Valencia_Sale'] = valencia_sale['Valencia_Sale']
    datasets['properties_by_district'] = properties_by_district['properties_by_district']

    return datasets


if __name__ == '__main__':
    download_rdata()
    datasets = read_data()

    # Guardar los datos en CSV en C:\Users\costa\Desktop\TFG\CSV
    CSV_FOLDER = r"C:\Users\costa\Desktop\TFG\1.0 Bajar datos en R y convertirlos a CSV\CSV Resultantes del Script"
    os.makedirs(CSV_FOLDER, exist_ok=True)  # Crear la carpeta si no existe

    for dataset_name, dataset in datasets.items():
        if isinstance(dataset, dict):  # Si el dataset es un diccionario
            for key, df in dataset.items():
                if isinstance(df, pd.DataFrame):  # Solo guardar si es un DataFrame
                    csv_path = os.path.join(CSV_FOLDER, f"{key}.csv")
                    df.to_csv(csv_path, index=False)  # Guardar sin �ndice
                    print(f"Guardado: {csv_path}")
        elif isinstance(dataset, pd.DataFrame):  # Si el dataset es un DataFrame directamente
            csv_path = os.path.join(CSV_FOLDER, f"{dataset_name}.csv")
            dataset.to_csv(csv_path, index=False)
            print(f"Guardado: {csv_path}")