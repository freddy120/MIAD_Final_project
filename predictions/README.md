## Predicciones

Se descarga el modelo entrenado del bucket y se realizan las predicciones, y finalmente se almacenan en base de datos.

```bash
cd ~/MIAD_Final_project/model
conda activate api
python3 predictions.py
```

## Logging finalización del script
Se usa Google Logging para mostrar la finalización de la ejecución del script y muestra los días predichos.


![](https://github.com/freddy120/MIAD_Final_project/blob/main/images/logging2.png)