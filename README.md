# Rainfall model skill assessment

This repo includes the code used in the project **Rainfall model skill assessment for AA**. You can find an extensive description of the project in [this presentation](https://docs.google.com/presentation/d/1TCKphDWAFWEWRGjlCnVkA087xuIdamsIfuGOIfOrDlo/edit#slide=id.g228236845b7).

## Usage

The Python version currently used is 3.9. Follow the steps to install [OCHA Anticipy](https://aa-toolbox.readthedocs.io/en/latest/quickstart.html). Then install the remaining dependencies from ``requirements.txt``:

```shell
pip install -r requirements.txt
```

To be able to [download data using the IRI API](https://aa-toolbox.readthedocs.io/en/latest/datasources/iri_seasonal_forecast.html), a key is required for authentication. The IRI API key should be stored in a variable called IRI_AUTH in a .env file in the main folder.

To be able to download ECMWF data, you should have a Copernicus account and Climate Data Store (CDS) credentials saved on your laptop. See [here](https://cds.climate.copernicus.eu/api-how-to) for details. 

The main code is contained in the notebook [main_notebook.ipynb](https://github.com/OCHA-DAP/pa-rainfall-model-skills/blob/main/main_notebook.ipynb).

