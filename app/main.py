from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import lda_module 
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/linear-discriminant-analysis")
def calculo(samples: int, features: int, classes: int):
    output_file = 'linear-discriminant-analysis.png'
    
    # Generar datos de ejemplo
    np.random.seed(0)
    n_samples = samples
    n_features = features
    n_classes = classes

    X = np.random.randn(n_samples, n_features)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    X[:n_samples // 2] += 1  # Desplazar la primera clase

    # Convertir datos a tensores de PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float64)
    y_tensor = torch.tensor(y, dtype=torch.float64)

    # Aplicar LDA
    result = lda_module.lda_fit_transform(X_tensor, y_tensor)
    X_lda = result["transformed"].numpy()
    components = result["components"].numpy()

    # Gráfico de dispersión de los datos originales
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f'Clase {int(label)}')
    plt.title('Datos Originales')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()

    # Gráfico de los datos transformados por LDA
    plt.subplot(1, 2, 2)
    for label in np.unique(y):
        plt.scatter(X_lda[y == label, 0], np.zeros_like(X_lda[y == label, 0]), label=f'Clase {int(label)}')
    plt.title('Datos Transformados por LDA')
    plt.xlabel('Componente Discriminante 1')
    plt.yticks([])  # Ocultar eje y
    plt.legend()

    plt.tight_layout()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/linear-discriminant-analysis-graphs")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
