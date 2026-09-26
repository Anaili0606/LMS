# LMS

# Entrenamiento de Compuertas Lógicas con DLMS

Este proyecto implementa el algoritmo **DLMS (Least Mean Squares)** para entrenar un modelo capaz de aproximar el comportamiento de las compuertas lógicas **AND, OR, NAND y NOR**.

Se realizan pruebas utilizando diferentes valores de la tasa de aprendizaje **μ (mu)**:

- μ = 0.10
- μ = 0.25
- μ = 0.50
- μ = 0.75

Para cada combinación de compuerta y valor de μ se registran:

- Pesos iniciales aleatorios.
- Pesos finales obtenidos después del entrenamiento.
- Salidas deseadas.
- Salidas aproximadas.
- Salidas binarias.
- Error promedio.
- Error cuadrático medio (MSE).
- Número de épocas utilizadas.
- Estado de convergencia.

El programa también genera gráficas del **descenso del error**, permitiendo observar el comportamiento del algoritmo durante el entrenamiento y comparar la convergencia para los diferentes valores de μ.

El entrenamiento utiliza una tolerancia de **0.001** y un máximo de **500 épocas**.
