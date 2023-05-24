
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
* Maria Paula Salamanca Delgado (202124610)
* Jorge Oswaldo Suárez Rodríguez (202124611)
* William Alexander Romero Bolivar (202124643)
* Freddy Rodrigo Mendoza Ticona (202120936)

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


## Resultados


## Entrenamiento continuo del modelo

## Tableros en Power BI



## Referencias:

[1] D. Salinas et al. High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes, International Journal of Forecasting (2019).