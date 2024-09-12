import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats


class AnalysisData:
    def __init__(self, filename):
        self.model = None
        self.data = filename
        self.data_procesada = None
        self.numeric_columns = []
        self.target_column = None

    def procesamiento(self):
        self.data_procesada = pd.read_csv(self.data)
        self.numeric_columns = self.data_procesada.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.target_column = self.data_procesada.columns[-1]
        self.data_procesada['TempIndex'] = range(len(self.data_procesada))
        print(self.data_procesada.describe())

    def prueba_normalidad(self):
        for column in self.numeric_columns:
            if self.data_procesada[column].nunique() > 1:
                stat, p = stats.normaltest(self.data_procesada[column])
                print(f'Prueba de normalidad para {column}:')
                print(f'Estadística={stat:.3f}, p-valor={p:.3f}')
                if p < 0.05:
                    print(f'{column} no parece ser normalmente distribuida (p<0.05)')
                else:
                    print(f'{column} parece ser normalmente distribuida (p>0.05)')
            else:
                print(f'{column} tiene un solo valor único. No se puede realizar la prueba de normalidad.')
            print()

    def grafica(self):
        # Visualización de la distribución de las características numéricas
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.data_procesada[self.numeric_columns])
        plt.title('Distribución de las características numéricas')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('distribucion_caracteristicas.png')
        plt.close()

        # Correlación entre variables numéricas
        correlation_matrix = self.data_procesada[self.numeric_columns].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de correlación')
        plt.tight_layout()
        plt.savefig('matriz_correlacion.png')
        plt.close()

        # Predicción simple usando regresión lineal
        X = self.data_procesada[self.numeric_columns[:-1]]  # Todas las columnas numéricas excepto la última
        y = self.data_procesada[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluación del modelo
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"R-squared en entrenamiento: {train_score}")
        print(f"R-squared en prueba: {test_score}")

        # Visualización de la predicción vs valores reales
        y_pred = model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title('Predicción vs Valores reales')
        plt.savefig('prediccion_vs_real.png')
        plt.close()

        # Evolución temporal de las características numéricas
        plt.figure(figsize=(12, 8))
        for column in self.numeric_columns[:-1]:  # Excluimos la columna objetivo
            plt.plot(self.data_procesada['TempIndex'], self.data_procesada[column], label=column)
        plt.xlabel('Índice temporal')
        plt.ylabel('Valor')
        plt.title('Evolución temporal de las características numéricas')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('evolucion_temporal.png')
        plt.close()

        print("Análisis completado. Se han generado gráficos en archivos PNG.")

    def crear_diagrama_torta(self):
        # Crear diagrama de torta para la variable objetivo
        plt.figure(figsize=(10, 8))
        target_counts = self.data_procesada[self.target_column].value_counts()
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f'Distribución de {self.target_column}')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.savefig('diagrama_torta.png')
        plt.close()

        # Si la variable objetivo es numérica, también crear un diagrama de torta para rangos
        if self.data_procesada[self.target_column].dtype in ['int64', 'float64']:
            plt.figure(figsize=(10, 8))
            bins = pd.cut(self.data_procesada[self.target_column], bins=5)
            bin_counts = bins.value_counts().sort_index()
            plt.pie(bin_counts.values, labels=[str(interval) for interval in bin_counts.index],
                    autopct='%1.1f%%', startangle=90)
            plt.title(f'Distribución de {self.target_column} (Agrupado)')
            plt.axis('equal')
            plt.savefig('diagrama_torta_agrupado.png')
            plt.close()

    def bosque_aleatorio(self):
        X = self.data_procesada[self.numeric_columns[:-1]]  # Todas las columnas numéricas excepto la última
        y = self.data_procesada[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de Bosque Aleatorio para clasificación
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Realizar predicciones
        y_pred = self.model.predict(X_test)

        # Generar el classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Imprimir el classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Crear una matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Visualización de la importancia de las características
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': self.model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Importancia de las características en el Bosque Aleatorio')
        plt.tight_layout()
        plt.savefig('importancia_caracteristicas.png')
        plt.close()
        print(feature_importance)
        # Guardar el modelo
        joblib.dump(self.model, 'modelo_bosque_aleatorio.joblib')
        print("Modelo de Bosque Aleatorio guardado como 'modelo_bosque_aleatorio.joblib'")

        return report