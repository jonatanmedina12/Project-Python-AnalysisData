# Análisis de Datos

Este proyecto realiza un análisis de datos sobre un conjunto de datos OCRA (Occupational Repetitive Actions), que contiene información sobre posturas y movimientos en entornos laborales.

## Requisitos

Para ejecutar este análisis, necesitarás Python 3.7+ y las siguientes bibliotecas:

- pandas
- matplotlib
- scikit-learn
- seaborn
- scipy
- pip install -r .\requirements.txt

Puedes instalar estas bibliotecas usando pip:

```
pip install pandas matplotlib scikit-learn seaborn scipy
```

## Descripción del Análisis

El script `data_analysis.py` realiza las siguientes tareas:

1. **Carga y Preprocesamiento de Datos**: 
   - Carga los datos del archivo CSV.
   - Crea nuevas características agregadas para hombros, codos, muñecas y manos.

2. **Análisis Descriptivo**:
   - Genera estadísticas descriptivas básicas de los datos.

3. **Prueba de Normalidad**:
   - Realiza una prueba de normalidad (D'Agostino y Pearson's) para las características principales y el índice de riesgo.

4. **Visualizaciones**:
   - Genera un diagrama de caja para mostrar la distribución de las características principales.
   - Crea una matriz de correlación para visualizar las relaciones entre variables.
   - Produce un gráfico de dispersión para comparar las predicciones con los valores reales.
   - Genera un gráfico de líneas para mostrar la evolución temporal de las características principales.

5. **Predicción**:
   - Utiliza regresión lineal para predecir el índice de riesgo (FRisk) basado en las características agregadas.

## Ejecución

Para ejecutar el análisis, simplemente corre el script:

```
python data_analysis.py
```

## Resultados

El script generará varios archivos PNG con visualizaciones:

- `distribucion_caracteristicas.png`: Muestra la distribución de las características principales.
- `matriz_correlacion.png`: Visualiza la correlación entre diferentes variables.
- `prediccion_vs_real.png`: Compara las predicciones del modelo con los valores reales.
- `evolucion_temporal.png`: Muestra cómo cambian las características principales a lo largo del tiempo.

Además, el script imprimirá en la consola:

- Estadísticas descriptivas de los datos.
- Resultados de las pruebas de normalidad.
- Métricas de rendimiento del modelo de regresión lineal.

## Interpretación de Resultados

- Las pruebas de normalidad te ayudarán a entender si las variables siguen una distribución normal, lo cual es importante para ciertos análisis estadísticos.
- La matriz de correlación te mostrará qué variables están más fuertemente relacionadas entre sí.
- El gráfico de predicción vs valores reales te dará una idea visual de qué tan bien está funcionando el modelo de regresión lineal.
- La evolución temporal te permitirá ver si hay patrones o tendencias a lo largo del tiempo en las diferentes medidas.

## Próximos Pasos

Este análisis es un punto de partida. Algunas ideas para expandir el análisis incluyen:

1. Realizar un análisis más detallado de las tendencias temporales.
2. Probar modelos de predicción más avanzados, como Random Forests o Gradient Boosting.
3. Realizar un análisis de componentes principales (PCA) para reducir la dimensionalidad de los datos.
4. Investigar más a fondo las relaciones entre las diferentes medidas y el riesgo (FRisk).

## Licencia
MIT License