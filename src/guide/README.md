# Explicación

"Mejora de la precisión en el modelo MobileNet mediante técnicas de fine-tuning y data augmentation"

## Introducción

En este proyecto, nos enfrentamos al desafío de mejorar la precisión de un modelo de clasificación basado en la arquitectura MobileNet.
La arquitectura MobileNet es ampliamente utilizada en tareas de clasificación de imágenes debido a su eficiencia y capacidad para funcionar en dispositivos con recursos limitados.

Sin embargo, aunque MobileNet ofrece resultados prometedores, siempre existe margen para mejorar su precisión. Para abordar este desafío, hemos explorado dos técnicas fundamentales: fine-tuning y data augmentation.

El fine-tuning implica ajustar los pesos de las capas de un modelo pre-entrenado para adaptarlo a un conjunto de datos específico, mientras que la data augmentation consiste en generar nuevas muestras de entrenamiento a partir de las existentes mediante transformaciones como rotaciones, brillo o ambos.

En esta presentación, compartiremos los resultados de nuestro estudio, destacando cómo la implementación de estas técnicas ha contribuido a mejorar la precisión del modelo MobileNet en la clasificación de imágenes. Además, analizaremos las matrices de confusión y los gráficos de entrenamiento para comprender mejor el impacto de estas técnicas en el rendimiento del modelo.

## Resultados del modelo MobileNet

Estos fueron los resultados obtenidos mediante la aplicación de técnicas de fine-tuning y data augmentation al modelo MobileNet. Los resultados se basan en la precisión alcanzada para cada combinación de data augmentation y fine-tuning en diferentes conjuntos de datos de prueba.

::: details Resultado D

|   Model   | Data Augmentation (DA) | Fine Tuning Layers (2) |      Fine Tuning Layers (4)      | Fine Tuning Layers (6) | Fine Tuning Layers (8) | Fine Tuning Layers (10) |
| :-------: | :--------------------: | :--------------------: | :------------------------------: | :--------------------: | :--------------------: | :---------------------: |
| MobileNet |    No Augmentation     |         0.6175         | <font color="blue">0.8900</font> |         0.8375         |         0.8525         |       **0.8800**        |
| MobileNet |        Flip HV         |         0.6200         |              0.8550              |         0.8500         |         0.8650         |       **0.8850**        |
| MobileNet |       Brightness       |         0.6175         |              0.8225              |         0.8350         |       **0.8825**       |         0.8650          |
| MobileNet |   Flip + Brightness    |         0.6350         |              0.8375              |         0.6750         |         0.8525         |         0.8675          |

<img :src="$withBase('/img/matriz.png')" class="center">

<img :src="$withBase('/img/max-pos.png')" class="center">

<img :src="$withBase('/img/min-neg.png')" class="center">
:::

<!-- |   Model   | Data Augmentation (DA) | Fine Tuning Layers | Accuracy |
|:---------:|:-----------------------:|:------------------:|:--------:|
| MobileNet |     No Augmentation     |         2          |  0.6175  |
| MobileNet |         Flip HV         |         2          |  0.6200  |
| MobileNet |       Brightness        |         2          |  0.6175  |
| MobileNet |   Flip + Brightness     |         2          |  0.6350  |
| MobileNet |     No Augmentation     |         4          |  0.8900  |
| MobileNet |         Flip HV         |         4          |  0.8550  |
| MobileNet |       Brightness        |         4          |  0.8225  |
| MobileNet |   Flip + Brightness     |         4          |  0.8375  |
| MobileNet |     No Augmentation     |         6          |  0.8375  |
| MobileNet |         Flip HV         |         6          |  0.8500  |
| MobileNet |       Brightness        |         6          |  0.8350  |
| MobileNet |   Flip + Brightness     |         6          |  0.6750  |
| MobileNet |     No Augmentation     |         8          |  0.8525  |
| MobileNet |         Flip HV         |         8          |  0.8650  |
| MobileNet |       Brightness        |         8          |  0.8825  |
| MobileNet |   Flip + Brightness     |         8          |  0.8525  |
| MobileNet |     No Augmentation     |        10          |  0.8800  |
| MobileNet |         Flip HV         |        10          |  0.8850  |
| MobileNet |       Brightness        |        10          |  0.8650  |
| MobileNet |   Flip + Brightness     |        10          |  0.8675  | -->

|   Model   | Data Augmentation (DA) | Fine Tuning Layers (2) | Fine Tuning Layers (4) | Fine Tuning Layers (6) |       Fine Tuning Layers (8)       | Fine Tuning Layers (10) |
| :-------: | :--------------------: | :--------------------: | :--------------------: | :--------------------: | :--------------------------------: | :---------------------: |
| MobileNet |    No Augmentation     |         0.6325         |         0.8500         |         0.7800         | <font color="blue"> 0.8800 </font> |         0.8650          |
| MobileNet |        Flip HV         |         0.6575         |         0.8675         |         0.8525         | <font color="blue"> 0.8800 </font> |         0.8625          |
| MobileNet |       Brightness       |         0.6375         |         0.8175         |         0.8225         |               0.8550               |         0.8675          |
| MobileNet |   Flip + Brightness    |         0.5975         |         0.8450         |         0.8350         |               0.8575               |         0.8625          |

### Primer resultado

<img :src="$withBase('/img/matriz1.png')" class="center">

<img :src="$withBase('/img/1.png')" class="center">

<img :src="$withBase('/img/2.png')" class="center">

### Segundo resultado

<img :src="$withBase('/img/matriz2.png')" class="center">

<img :src="$withBase('/img/2-2.png')" class="center">

<img :src="$withBase('/img/3-2.png')" class="center">

## Comparación de Resultados

### Primer Conjunto de Resultados

1. **Matriz de confusión**:

   - Verdaderos negativos: 195
   - Falsos positivos: 5
   - Falsos negativos: 43
   - Verdaderos positivos: 157

2. **Exactitud (accuracy)**: 0.88

3. **Rendimiento**:
   - **Precisión**:
     - Entrenamiento: Se estabiliza cerca de 1.0.
     - Validación: Alcanza un máximo alrededor de 0.82.
   - **Pérdida**:
     - Entrenamiento: Disminuye continuamente hasta casi 0.0.
     - Validación: Se mantiene alrededor de 0.5.

### Segundo Conjunto de Resultados

1. **Matriz de confusión**:

   - Verdaderos negativos: 184
   - Falsos positivos: 16
   - Falsos negativos: 32
   - Verdaderos positivos: 168

2. **Exactitud (accuracy)**: 0.88

3. **Rendimiento**:
   - **Precisión**:
     - Entrenamiento: Se estabiliza cerca de 0.9.
     - Validación: Alcanza un máximo alrededor de 0.85.
   - **Pérdida**:
     - Entrenamiento: Disminuye continuamente hasta casi 0.1.
     - Validación: Se mantiene alrededor de 0.4-0.5.

### Resumen de la Comparación

- **Exactitud**: Ambos conjuntos tienen una exactitud del 88%.
- **Matriz de confusión**:
  - El primer conjunto tiene menos falsos positivos (5 vs 16) pero más falsos negativos (43 vs 32).
  - El segundo conjunto tiene un mejor equilibrio entre falsos positivos y falsos negativos.
- **Rendimiento**:
  - La precisión de la validación es ligeramente mejor en el segundo conjunto (0.85 vs 0.82).
  - La pérdida de validación es similar en ambos conjuntos (alrededor de 0.5).

En general, aunque ambos conjuntos tienen la misma exactitud del 88%, el segundo conjunto de resultados muestra un equilibrio más adecuado entre falsos positivos y falsos negativos, y una precisión de validación ligeramente mejor.

## Hallazgos y Conclusiones

**Resumen de los hallazgos**:

- Se utilizó la técnica de fine-tuning y data augmentation para mejorar la precisión del modelo MobileNet.
- Ambos conjuntos de resultados alcanzaron una exactitud del 88%.
- El segundo conjunto de resultados mostró un mejor equilibrio entre falsos positivos y falsos negativos, y una precisión de validación ligeramente superior.

**Conclusiones sobre la efectividad**:

- **Fine-tuning**: Ajustar las capas superiores de MobileNet permitió al modelo adaptarse mejor a las características específicas del conjunto de datos, mejorando su capacidad para distinguir entre etiquetas benignas y malignas.
- **Data augmentation**: Incrementar la variabilidad en los datos de entrenamiento mediante técnicas como rotación, desplazamiento y cambio de brillo ayudó a evitar el sobreajuste y a mejorar la robustez del modelo.

**Recomendaciones para futuros trabajos**:

- **Explorar más técnicas de data augmentation**: Investigar métodos adicionales como el corte aleatorio, la mezcla de imágenes, o el uso de GANs para generar ejemplos sintéticos.
- **Experimentar con diferentes arquitecturas de modelos**: Probar otras redes neuronales preentrenadas como EfficientNet o ResNet para comparar su rendimiento.
- **Optimizar el proceso de fine-tuning**: Ajustar parámetros como la tasa de aprendizaje y el número de capas congeladas para encontrar el balance óptimo entre adaptación y preservación de características preentrenadas.

## Propuestas de Preprocesamiento

**Métodos de preprocesamiento sugeridos**:

1. **Pre-segmentación**:

   - **Descripción**: Utilizar técnicas de segmentación de imágenes para aislar las regiones de interés antes de pasar las imágenes al modelo de clasificación.
   - **Ejemplo**: Aplicar segmentación basada en contornos o segmentación semántica para enfocar el análisis en áreas específicas de las imágenes, como las lesiones cutáneas.

2. **Filtros de imagen**:

   - **Descripción**: Aplicar filtros para mejorar la calidad y las características relevantes de las imágenes.
   - **Ejemplo**: Uso de filtros de realce de bordes, suavizado, o eliminación de ruido para mejorar la claridad de las características diagnósticas.

3. **Técnicas de aumento de datos específicas**:
   - **Descripción**: Implementar técnicas de aumento de datos que sean particularmente relevantes para el dominio del problema.
   - **Ejemplo**: Aplicar transformaciones geométricas específicas que imiten variaciones naturales en las imágenes médicas, como cambios en la iluminación y la orientación del tejido.

## Otros Modelos

### Data Augmentation

| Model     | Data Augmentation | Fine-tuning | Accuracy                           |
| --------- | ----------------- | ----------- | ---------------------------------- |
| MobileNet | Without DA        | 2 layers    | 0.8600                             |
| MobileNet | DA                | 2 layers    | 0.8275                             |
| MobileNet | Without DA        | 4 layers    | 0.7800                             |
| MobileNet | DA                | 4 layers    | <font color="blue"> 0.8850 </font> |
| ResNet50  | Without DA        | 2 layers    | 0.5025                             |
| ResNet50  | DA                | 2 layers    | 0.7000                             |
| ResNet50  | Without DA        | 4 layers    | 0.6000                             |
| ResNet50  | DA                | 4 layers    | 0.5050                             |

<img :src="$withBase('/img/predes1.png')" class="center">

### Models

|    Model    | Fine-tuning |              Accuracy              |
| :---------: | :---------: | :--------------------------------: |
|  MobileNet  |  2 layers   |               0.9000               |
|  MobileNet  |  4 layers   |               0.8925               |
|  MobileNet  |  6 layers   |               0.8925               |
|  MobileNet  |  8 layers   |               0.8950               |
|  MobileNet  |  10 layers  |               0.8850               |
|  ResNet50   |  2 layers   |               0.6000               |
|  ResNet50   |  4 layers   |               0.7050               |
|  ResNet50   |  6 layers   |               0.8925               |
|  ResNet50   |  8 layers   |               0.6625               |
|  ResNet50   |  10 layers  |               0.7575               |
| InceptionV3 |  2 layers   |               0.8825               |
| InceptionV3 |  4 layers   |               0.8725               |
| InceptionV3 |  6 layers   |               0.8700               |
| InceptionV3 |  8 layers   |               0.8775               |
| InceptionV3 |  10 layers  |               0.8675               |
| DenseNet121 |  2 layers   |               0.8800               |
| DenseNet121 |  4 layers   |               0.8925               |
| DenseNet121 |  6 layers   |               0.8825               |
| DenseNet121 |  8 layers   | <font color="blue"> 0.9025 </font> |
| DenseNet121 |  10 layers  | <font color="blue"> 0.9025 </font> |

<img :src="$withBase('/img/resultado1.png')" class="center">
<img :src="$withBase('/img/resultado2.png')" class="center">

DenseNet121 es una arquitectura de red neuronal conocida por su eficiencia y rendimiento en tareas de visión por computadora. 
<!-- Aumentar el número de capas en una red neuronal generalmente aumenta su capacidad de aprendizaje, pero puede llevar a problemas de sobreajuste si no se maneja correctamente. En este caso, agregar más capas (de 8 a 10) a DenseNet121 no ha mejorado ni empeorado significativamente su precisión (0.9025), lo que sugiere que el conjunto de datos puede no ser lo suficientemente complejo como para aprovechar las capas adicionales o que el modelo ya está bien entrenado con las 8 capas. -->


<!-- El alto rendimiento de DenseNet121 en comparación con otros modelos puede atribuirse a varias razones: -->

- Arquitectura eficiente: DenseNet121 es una arquitectura de red neuronal convolucional (CNN) diseñada para maximizar la eficiencia y el rendimiento. 

- Transferencia de aprendizaje: DenseNet121 a menudo se preentrena en conjuntos de datos masivos, como ImageNet, antes de ser ajustado a tareas específicas. Esta preentrenamiento permite que el modelo adquiera un conocimiento general sobre una amplia gama de características visuales, lo que puede beneficiar el rendimiento en tareas más específicas, como la clasificación de imágenes.

- Regularización efectiva: A pesar de tener un gran número de parámetros, DenseNet121 está diseñado con técnicas efectivas de regularización, como la normalización por lotes y la regularización L2. Estas técnicas ayudan a prevenir el sobreajuste al limitar la complejidad del modelo y mejorar su capacidad de generalización.

- Optimización del entrenamiento: Los detalles específicos del entrenamiento, como la tasa de aprendizaje, el tamaño del lote y la función de pérdida utilizada, pueden influir significativamente en el rendimiento de un modelo. DenseNet121 puede haber sido entrenado con una configuración óptima de estos hiperparámetros, lo que contribuye a su alto rendimiento.

En resumen, el alto rendimiento de DenseNet121 puede atribuirse a su arquitectura eficiente, transferencia de aprendizaje, técnicas de regularización efectivas y optimización del entrenamiento. Estos factores combinados pueden haber contribuido a que DenseNet121 supere a otros modelos en términos de precisión en el conjunto de datos específico en el que se evaluó.





