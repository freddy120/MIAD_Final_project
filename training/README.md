## Entrenamiento

Primero se hace validación, es decir, se separa una muestra de validación para calcular las métricas de desempeño (MSE, MAE, RMSE, MAPE, SMAPE) con validacion.py y luego se entrena el modelo con todos los datos disponibles en train_cpu.py.

```bash
cd /home/freddy_12_120/MIAD_Final_project/training/
conda activate api
python3 validation.py
python3 train_cpu.py
```

## Logging de métricas y finalización de los scripts
Se usa Google Logging para mostrar resultados y poder monitorear el performance del modelo con el paso del tiempo.

métricas por serie de tiempo:

![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/logging1.png)

finalización de la validación y entrenamiento:
![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/logging2.png)

## Guardado de artefactos del modelo
El modelo entrenado se guarda en google cloud storage para luego ser descargado en la etapa de predicción.


Se guarda con la fecha de entrenamiento:

![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/bucket1.png)

Son 4 objetos del modelo necesarios para las predicciones.

![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/bucket2.png)