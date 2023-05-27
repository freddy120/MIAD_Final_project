
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/miad.jpg)


# Proyecto final de la Maestría en Inteligencia Analítica de Datos MIAD

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![Build Status](https://travis-ci.org/anfederico/clairvoyant.svg?branch=master)](https://travis-ci.org/freddy120/MIAD_no_supervisado_project)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![GitHub Issues](https://img.shields.io/github/issues/freddy120/MIAD_no_supervisado_project.svg)](https://github.com/freddy120/MIAD_no_supervisado_project/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Titulo: Visualización de datos históricos y predicción del índice S&P500 y variables relevantes

## Integrantes
* Maria Paula Salamanca Delgado (202124610) m.salamancad@uniandes.edu.co
* Jorge Oswaldo Suárez Rodríguez (202124611) j.suarez1@uniandes.edu.co
* William Alexander Romero Bolivar (202124643) w.romerob@uniandes.edu.co
* Freddy Rodrigo Mendoza Ticona (202120936) r.mendozat@uniandes.edu.co

## Resumen
La idea del proyecto es poder visualizar los valores históricos del índice S&P500 junto con las principales acciones que lo componen y con las variables e indicadores económicos que influyen en su variación, para que nos ayude a encontrar relaciones entre estos, además se construye un modelo de múltiples series de tiempo para la predicción principalmente del índice S&P500 en base a sus valores pasados como también de los valores pasados de las principales acciones y de las variables/indicadores económicos más relevantes, de esta forma el modelo nos podrá predecir el valor futuro de las acciones y del índice S&P 500 que nos sirve como soporte para la toma de decisiones de inversión.
El modelo usado es DeepVAR [1], el cual es un modelo multivariado (multivariate time-series forecasting) que nos permite predecir todas las series de tiempo con un solo modelo considerando las relaciones entre ellas.

DeepVAR ofrece las siguientes ventajas para el pronóstico de series de tiempo multivariado:
* Flexibilidad: DeepVAR puede manejar conjuntos de datos con una gran cantidad de variables, capturando relaciones complejas y dependencias entre ellas.
* Modelado de dependencia temporal: las capas recurrentes en DeepVAR están diseñadas para capturar dependencias temporales en varios lapsos de tiempo, lo que permite que el modelo aprenda de patrones históricos.
* Embeddings de variable: la capa de incorporación permite que cada variable tenga su propia representación vectorial continua, lo que ayuda al modelo a aprender características únicas para cada variable.
* Codificación automática: el mecanismo de codificación automática anima al modelo a aprender una representación compacta de la secuencia de entrada, lo que permite la generalización y la robustez frente al ruido.

## Arquitectura

Se hace uso de los siguientes recursos de GCP:
* 1 instancia e2-standard-2 (2 vCPU, 8GB RAM) con Ubuntu 20.04 (Compute engine).
* 1 instancia SQL con postgresql (SQL).
* 1 bucket (Cloud storage).
* Logging de métricas (Logging).


![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/architecture.svg)

La instancia de cómputo tiene como funciones:
* Realizar el ETL que consiste en descargar los datos históricos y almacenarlos en la base de datos.
* Realizar la validación y entrenamiento del modelo, y como resultado guardar el modelo en el bucket.
* Realizar las predicciones haciendo uso del modelo guardado en el bucket.

La instancia SQL tiene como funciones:
* Almacenar en tablas los datos históricos y predicciones de las series de tiempo.
* Se usa como fuente de datos para Power BI.

El bucket tiene como funciones:
* Almacenar los artefactos del modelo entrenado.
* Almacenar un histórico de los modelos entrenados.

El logging tiene como funciones:
* Monitoreo del desempeño del modelo con el paso del tiempo.
* Monitoreo del entrenamiento diario del modelo y la realización de predicciones.


## Datos
Se extraen los datos de Yahoo Finance y Federal Reserve Economic Data (FRED) a través de los paquetes de Python yfinance y pandas-datareader. Los datos son livianos, de acceso inmediato y se descargan de internet en cualquier momento mediantes las APIs a las que se acceden a través de las librerías de Python.

Se trabaja con los datos desde el 2012 a la actualidad.

Las variables descargadas son las siguientes:

Yahoo Finance:

| Variable | Descripción|                                              
| --------- | ------------------------------------------------------------ |
| ^GSPC     | S&P500                                                      |
| AAPL      | Apple Inc.                                                    |
| MSFT      | Microsoft Corporation                                         |
| GOOG      | Alphabet Inc.                                                 |
| GOOGL     | Alphabet Inc.                                                 |
| TSLA      | Tesla, Inc.                                                   |
| AMZN      | Amazon.com, Inc.                                              |
| BRK-A     | Berkshire Hathaway Inc.                                       |
| BRK-B     | Berkshire Hathaway Inc.                                       |
| NVDA      | NVIDIA Corporation                                            |
| META      | Meta Platforms, Inc.                                          |
| UNH       | UnitedHealth Group Inc.                                       |
| BZ=F      | Brent Crude Oil                                               |
| NG=F      | Natural Gas                                                   |
| GC=F      | Gold                                                          |
| EURUSD=X  | EUR/USD                                                       |
| ^VIX      | CBOE Volatility Index                                         |
| ^IXIC     | NASDAQ Composite                                              |



Federal Reserve Economic Data (FRED):

| Variable | Descripción|   
| --------- | ------------------------------------------------------------ |
| EFFR      | Effective Federal Funds Rate                                  |
| GDP       | Gross Domestic Product                                        |
| CSUSHPISA | S&P/Case-Shiller U.S. National Home Price Index                |
| CPIAUCSL  | Consumer Price Index for All Urban Consumers: All Items in U.S. City Average |
| CPILFESL  | Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average |


## ETL
Se descargan y guardan en base de datos los datos históricos de las variables de interés.

Se descarga los datos desde el 2012 a la actualidad.

```python
# ...
import yfinance
import yfinance as yf
import pandas_datareader as pdr
import pandas_datareader.data as web
# ...

client_logging = google.cloud.logging.Client()
client_logging.setup_logging()

# ...

date_start = '2012-01-01'
date_end = datetime.datetime.now().strftime('%Y-%m-%d')


print(f"Downloading data from:{date_start} to:{date_end}")

tickers = ["^GSPC", 'AAPL', 'MSFT','GOOG','GOOGL','TSLA','AMZN','BRK-A','BRK-B','NVDA','META','UNH','BZ=F','NG=F', 'GC=F', 'EURUSD=X','^VIX','^IXIC']

from_date = None
to_date = None

for ticker in tickers:
    print(f"Downloading data for ticker: {ticker}")
    stock_data = yfinance.download(ticker, start=date_start, end=date_end)
    stock_data = stock_data.dropna()
    df = stock_data.reset_index().rename(columns={'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close':'adj_close', 'Volume':'volume'})
    from_date = df['date'].min()
    to_date = df['date'].max()

    table_name = "ts_" + re.sub(r'\W+', '', ticker).lower() # remove special characters from ticker and lower case

    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
    conn.commit()
    print(f"Wrote data to Table {table_name}, rows: {df.shape[0]}")


fred_labels = ['EFFR', 'CSUSHPISA', 'GDP', 'CPIAUCSL', 'CPILFESL']

for label in fred_labels:
    print(f"Downloading data for ticker: {label}")
    fred_data = web.DataReader(label, 'fred', datetime.datetime.strptime(date_start, '%Y-%m-%d'), datetime.datetime.strptime(date_end, '%Y-%m-%d'))
    fred_data[label].fillna(method='ffill', inplace=True)
    fred_data = fred_data.dropna()
    fred_data.index.names = ['Date']
    df = fred_data.reset_index().rename(columns={'Date':'date', label:'value'})

    table_name = "ts_" + re.sub(r'\W+', '', label).lower() # remove special characters from ticker and lower case

    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
    conn.commit()
    print(f"Wrote data to Table {table_name}, rows: {df.shape[0]}")

# ...
```


## Ajuste de hiperparametros
Se hace uso de optuna para ajustar el modelo DeepVAR, se ajusta el learning rate, hidden size de capas RNN, el rank de la función de pérdida “Multivariate Normal Distribution Loss” y número de capas RNN del modelo DeepVAR.

```python
# ...
def tuning():
    # Download Data

    date_start = '2012-01-01'
    date_end = datetime.datetime.now().strftime('%Y-%m-%d')

    # ...

    # feature selection
    # ...

    # Transform data

    # ...

    # DeepVAR model
    max_prediction_length = 10
    max_encoder_length = 30

    training_cutoff = train_series["time_idx"].max() - max_prediction_length

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    global training
    training = TimeSeriesDataSet(
        train_series[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(train_series.series)},
        group_ids=["series"],
        static_categoricals=[
            "series"
        ], 
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )


    # for validation
    validation = TimeSeriesDataSet.from_dataset(training, train_series, min_prediction_idx=training_cutoff + 1)

    batch_size = 128

    global train_dataloader
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    global val_dataloader
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)


    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    with open("hyperparameters.json", "w") as file:
        json.dump(trial.params, file)

    print("DONE")


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
    hidden_size = trial.suggest_int("hidden_size", 4, 128)
    rank = trial.suggest_int("rank", 4, 30)
    rnn_layers = trial.suggest_int("rnn_layers", 1, 3)

    # Create the model
    model =  DeepAR.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        optimizer="Adam",
        loss=MultivariateNormalDistributionLoss(rank=rank),
    )
    
    
    # Create the PyTorch Lightning Trainer
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        gradient_clip_val=0.1,
        limit_train_batches=50,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10)],
    )
    

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    

# ...

```

## Entrenamiento del modelo de predicción
    
El modelo entrenado es el DeepVAR con los siguientes parametros:

```python
 net = DeepAR.from_dataset(
        training,
        learning_rate=0.006431956578942552,
        log_interval=10,
        log_val_interval=1,
        hidden_size=38,
        rnn_layers=1,
        optimizer="Adam",
        loss=MultivariateNormalDistributionLoss(rank=14),
    )

```


## Resultados
El modelo DeepVAR alcanzo métricas sobre 10 días de predicción del S&P500

```bash
MAPE: 0.5576595243631158
SMAPE: 0.560811992647919
MAE: 23.265193543142686
MSE: 1144.3109246677573
RMSE: 33.827665078567826
```

métricas de la prediccion de las principales acciones:

AAPL
```bash
MAPE: 2.1618667523354764
SMAPE: 2.191073695206338
MAE: 3.753178293256178
MSE: 17.41960204897644
RMSE: 4.173679677332275
```

AMZN
```bash
MAPE: 6.11248366164213
SMAPE: 6.338948341005639
MAE: 6.961216149168331
MSE: 57.4636532318337
RMSE: 7.580478430272967
```

GOOG
```bash
MAPE: 9.161140762251685
SMAPE: 9.639052809467067
MAE: 10.968042574762183
MSE: 131.39183665073682
RMSE: 11.46262782483741
```

GOOGL
```bash
MAPE: 9.275758398440221
SMAPE: 9.764506624519665
MAE: 11.056456823017403
MSE: 133.06969713764593
RMSE: 11.535583953040518
```

META
```bash
MAPE: 3.230670378149765
SMAPE: 3.313936607238387
MAE: 7.813054025453627
MSE: 95.91139843764543
RMSE: 9.793436497861485
```

MSFT
```bash
MAPE: 2.4845972349559404
SMAPE: 2.5236729082201066
MAE: 7.795505159508207
MSE: 76.24170120603812
RMSE: 8.731649397796394
```

NVDA
```bash
MAPE: 4.036488545430314
SMAPE: 4.22312114143283
MAE: 12.3692238457364
MSE: 342.5030032830427
RMSE: 18.50683666332641
```

TSLA
```bash
MAPE: 1.952950837076913
SMAPE: 1.9797689200362754
MAE: 3.403795401072389
MSE: 20.443587625878216
RMSE: 4.52145857283667
```

Además, con DeepVAR, como es un modelo probabilístico que modela la distribución de los valores futuros de cada serie de tiempo, podemos calcular los intervalos de confianza de las predicciones en intervalos de 50, 90, 95 y 98 porciento.

S&P500
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/sp500_pred.png)


AAPL
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/aapl_pred.png)

AMZN
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/amzn_pred.png)

GOOG
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/goog_pred.png)

GOOGL
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/googl_pred.png)

META
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/meta_pred.png)

MSFT
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/msft_pred.png)

NVDA
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/nvda_pred.png)

TSLA
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/tsla_pred.png)



## Entrenamiento continuo del modelo
Debido a que la serie de tiempo tiene periodicidad diaria y es volátil tenemos que entrenar el modelo diariamente para tener precisión en las predicciones futuras (a corto plazo y largo plazo), se configura un job en la instancia de GCP para que se entrene diariamente el modelo con los datos actualizados.

Job:
Se ejecuta el ETL, validación, entrenamiento del modelo y predicciones usando el modelo entrenado.

```bash
cd /home/freddy_12_120/MIAD_Final_project/etl/
conda activate etl
python3 etl_first.py
conda deactivate

cd /home/freddy_12_120/MIAD_Final_project/training/
conda activate api
python3 validation.py
python3 train_cpu.py

cd /home/freddy_12_120/MIAD_Final_project/predictions/
python3 predictions.py
conda deactivate

```


## Tableros en Power BI

El tablero de control desarrollado tiene como objetivo apoyar la toma de decisiones por parte de la Mesa de Dinero del Grupo Stanley (GS) al contar con visualizaciones del comportamiento histórico y de la predicción confiable a corto plazo (de uno a diez días) del S&P500 y variables de interés. A continuación se listan las instrucciones para su uso.

Se puede interactuar con el tablero de control [aqui](https://app.powerbi.com/view?r=eyJrIjoiYmU5ZGIwZmEtZTAwYS00ZmEwLWE2YjUtZTFkNTM2Y2VjZDVkIiwidCI6IjFlYmE0NDNmLTIzZTUtNDUzNC05MGQxLTA5NzZhYWJlODZhYyIsImMiOjR9)

**Instrucciones**:

Diariamente, antes del horario de apertura del mercado colombiano (7:00 am), se cargan automáticamente en la nube los datos históricos y la predicción del S&P500 al último día hábil anterior. Para acceder a los datos más actualizados es necesario actualizar la consulta dando clic en "Actualizar", en la pestaña "Inicio".

Para visualizar el comportamiento del S&P500 y variables de interés, ir a la pestaña “Principal”:

* Si se desea observar el comportamiento histórico para una variable específica en una ventana de tiempo determinada, se debe:

    * Hacer clic sobre el botón “Histórico” que se encuentra en la esquina superior izquierda (clic en la versión pública o Ctrl+clic en la versión de escritorio)
    * Seleccionar la acción o variable que se desea observar en el filtro “Activo”.
    * Deslizar el filtro de fecha hasta seleccionar la ventana de tiempo que se desea. 

* Si se desea observar la predicción para una variable específica en una ventana de tiempo determinada para cierto intervalo de confianza, se debe:
    * Hacer clic sobre el botón “Predicciones” que se encuentra en la esquina superior izquierda (clic en la versión pública o Ctrl+clic en la versión de escritorio)
    * Seleccionar la acción o variable que se desea observar en el filtro “Activo”
    * Deslizar el filtro de fecha hasta la ventana de tiempo que se desea. 
    * Seleccionar el intervalo de confianza que se desea observar en el filtro “Intervalo de confianza”. Por defecto muestra el intervalo de confianza correspondiente al 99%.
        * Para observar únicamente la predicción, seleccionar únicamente uno de los cuatro intervalos de confianza.
        * Para observar la predicción acompañada de los datos históricos, seleccionar uno de los intervalos de confianza y la opción “observado” (Ctrl+ clic).

Para visualizar relaciones relevantes en el comportamiento del S&P500 y otras variables, ir a la pestaña "Correlaciones":

* Si se desea observar la relación entre variables del mercado bursátil, se debe:
    * Hacer clic sobre el botón “Mercado bursátil” que se encuentra en la esquina superior izquierda (clic en la versión pública o Ctrl+clic en la versión de escritorio)
    * Seleccionar las acciones o variables que se desea observar en los filtros “Activo1” y “Activo2”.
    * Deslizar el filtro de fecha hasta seleccionar la ventana de tiempo que se desea. 

* Si se desea observar la relación entre variables del mercado bursátil y variables de macroeconómicas, se debe
    * Hacer clic sobre el botón “Variables macroeconómicas” que se encuentra en la esquina superior izquierda (clic en la versión pública o Ctrl+clic en la versión de escritorio)
    * Seleccionar las acciones o variables que se desea observar en los filtros “Activo1” y “Variable2”.
    * Deslizar el filtro de fecha hasta seleccionar la ventana de tiempo que se desea. 

## Referencias:

[1] D. Salinas et al. High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes, International Journal of Forecasting (2019).
