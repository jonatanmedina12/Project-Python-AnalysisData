from src.controller.analysis_data import AnalysisData


def execute():
    path=""
    exec_=AnalysisData(path)
    exec_.procesamiento()
    exec_.prueba_normalidad()
    exec_.grafica()
