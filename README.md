El **Análisis Discriminante Lineal (LDA)** es una técnica de aprendizaje supervisado utilizada para resolver problemas de clasificación, especialmente cuando se trata de múltiples clases. Su objetivo principal es encontrar combinaciones lineales de variables predictoras que maximicen la separación entre las diferentes clases, facilitando así la reducción de dimensionalidad y mejorando la interpretabilidad de los datos.

En el algoritmo LDA que desarrollé, se implementa el cálculo en C++ utilizando la biblioteca PyTorch y se expone la funcionalidad a Python mediante pybind11. El flujo general del algoritmo es el siguiente:

1. **Cálculo de las medias por clase y la media global**: Se calcula el vector de medias para cada clase y la media general de todas las muestras.

2. **Cálculo de las matrices de dispersión intra-clase (Sw) e inter-clase (Sb)**: Estas matrices miden la variabilidad dentro de cada clase y entre las clases, respectivamente.

3. **Resolución del problema de autovalores y autovectores**: Se resuelve la ecuación generalizada para obtener los autovectores que maximizan la separación entre clases.

4. **Transformación de los datos originales**: Se proyectan los datos en el nuevo espacio reducido utilizando los autovectores seleccionados.

Una vez realizada la transformación en C++, se utiliza un script en Python para visualizar los resultados como microservicio utilizando la librería Fastapi. Este script genera una gráfica de dispersión de las primeras dos componentes discriminantes, permitiendo observar la separación entre las clases en el nuevo espacio. Además, se puede incluir una gráfica adicional, como un gráfico de varianza explicada, para entender la proporción de varianza capturada por cada componente.

Este enfoque combina la eficiencia del cálculo en C++ con la flexibilidad y facilidad de visualización de Python, proporcionando una herramienta robusta para el análisis y clasificación de datos mediante LDA. 
