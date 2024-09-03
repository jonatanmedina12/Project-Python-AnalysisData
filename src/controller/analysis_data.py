import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats

class AnalysisData:
    def __init__(self, filename):
        self.data = filename
        self.data_procesada = None

    def procesamiento(self):
        self.data_procesada = pd.read_csv(self.data)

        # Crear un índice temporal
        self.data_procesada['TempIndex'] = range(len(self.data_procesada))

        # Resto del procesamiento...
        self.data_procesada['TotalShoulder'] = self.data_procesada['shoulderR_pose1'] + self.data_procesada['shoulderR_pose2'] + self.data_procesada['shoulderR_pose3'] + \
                                               self.data_procesada['shoulderL_pose1'] + self.data_procesada['shoulderL_pose2'] + self.data_procesada['shoulderL_pose3']
        self.data_procesada['TotalElbow'] = self.data_procesada['elbowR_pose1'] + self.data_procesada['elbowR_pose2'] + \
                                            self.data_procesada['elbowL_pose1'] + self.data_procesada['elbowL_pose2']
        self.data_procesada['TotalWrist'] = self.data_procesada['wristR_pose1'] + self.data_procesada['wristR_pose2'] + \
                                            self.data_procesada['wristL_pose1'] + self.data_procesada['wristL_pose2']
        self.data_procesada['TotalHand'] = self.data_procesada['handR_pose1'] + self.data_procesada['handR_pose2'] + \
                                           self.data_procesada['handR_pose3'] + \
                                           self.data_procesada['handL_pose1'] + self.data_procesada['handL_pose2'] + \
                                           self.data_procesada['handL_pose3']
        # Análisis descriptivo
        print(self.data_procesada.describe())

    def prueba_normalidad(self):
        # Prueba de normalidad
        variables = ['TotalShoulder', 'TotalElbow', 'TotalWrist', 'TotalHand', 'FRisk']
        for var in variables:
            if self.data_procesada[var].nunique() > 1:  # Solo realizar la prueba si hay más de un valor único
                stat, p = stats.normaltest(self.data_procesada[var])
                print(f'Prueba de normalidad para {var}:')
                print(f'Estadística={stat:.3f}, p-valor={p:.3f}')
                if p < 0.05:
                    print(f'{var} no parece ser normalmente distribuida (p<0.05)')
                else:
                    print(f'{var} parece ser normalmente distribuida (p>0.05)')
            else:
                print(f'{var} tiene un solo valor único. No se puede realizar la prueba de normalidad.')
            print()

    def grafica(self):
        # Visualización de la distribución de las características principales
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.data_procesada[['TotalShoulder', 'TotalElbow', 'TotalWrist', 'TotalHand']])
        plt.title('Distribución de las características principales')
        plt.savefig('distribucion_caracteristicas.png')
        plt.close()

        # Correlación entre variables
        correlation_matrix = self.data_procesada[
            ['TotalShoulder', 'TotalElbow', 'TotalWrist', 'TotalHand', 'FDura', 'FRecu', 'FFreqR', 'FFreqL', 'FForcR',
             'FForcL', 'FPostR', 'FPostL', 'FRisk']].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de correlación')
        plt.savefig('matriz_correlacion.png')
        plt.close()

        # Predicción simple usando regresión lineal
        X = self.data_procesada[['TotalShoulder', 'TotalElbow', 'TotalWrist', 'TotalHand']]
        y = self.data_procesada['FRisk']

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

        # Evolución temporal de las características principales
        plt.figure(figsize=(12, 8))
        for column in ['TotalShoulder', 'TotalElbow', 'TotalWrist', 'TotalHand']:
            plt.plot(self.data_procesada['TempIndex'], self.data_procesada[column], label=column)
        plt.xlabel('Índice temporal')
        plt.ylabel('Valor')
        plt.title('Evolución temporal de las características principales')
        plt.legend()
        plt.tight_layout()
        plt.savefig('evolucion_temporal.png')
        plt.close()

        print("Análisis completado. Se han generado gráficos en archivos PNG.")