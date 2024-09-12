from src.controller.analysis_data import AnalysisData


def execute():
    analisis = AnalysisData(r"C:\Users\Jonat\Downloads\diabetes.csv")
    analisis.procesamiento()
    analisis.prueba_normalidad()
    analisis.grafica()
    analisis.crear_diagrama_torta()
    report = analisis.bosque_aleatorio()
    print(f"Precisión promedio: {report['accuracy']:.2f}")
