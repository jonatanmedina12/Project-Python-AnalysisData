from src.controller.analysis_data import AnalysisData


def execute():
    path="D:\\Repositorios\\Project-Python-AnalysisData\\OCRA_raw (3).csv"
    exec_=AnalysisData(path)
    exec_.procesamiento()
    exec_.prueba_normalidad()
    exec_.grafica()
