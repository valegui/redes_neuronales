# Tarea 1
Repositorio para la Tarea 1 del curso Redes Neuronales y Programación Genética

Lenguaje utilizado: Python 3.7

# Implementación

En este directorio se encuentra la implementación para una red neuronal.

### Neuronas
La implementación de neuronas se realiza con la clase [``Perceptron``](./Perceptrons/Perceptron.py).

Posee las variables _bias_, _weights_, _output_, _delta_, _lr_ (learning rate) y _activation_ (función de activación).

En el caso de querer iniciar un objeto de la clase ``Perceptron`` sin pesos específicos, se debe indicar el número de inputs esperados.

Si bien esta implementación posee los métodos _train_ y _train\_all_, estos solo se utilizan cuando se quiere que una neurona aprenda individualmente y no como parte de una capa ni de una red neuronal, pues esto se hace a través de llamadas realizadas por las capas y la red.

### Red Neuronal
Para implementar la red neuronal se utilizan las clases [``NeuronLayer``](./Perceptrons/NeuronLayer.py) y [``NeuralNetwork``](./Perceptrons/NeuralNetwork.py), que representan una capa de perceptrones y una red neuronal respectivamente.

La clase [``NeuronLayer``](./Perceptrons/NeuronLayer.py) posee las variables _n_ (número de neuronas en la capa), _is\_output_, _next\_layer_ (siguiente capa en la red), _prev\_layer_ (capa anterior en la red) y _neurons_ (una lista de _n_ neuronas).

La clase [``NeuralNetwork``](./Perceptrons/NeuralNetwork.py) posee las variables _f_ (dimension de los inputs), _n_layers_ (número de capas escondidas más uno), _layers_ (una lista de capas de neuronas).

### Funciones de activación

Se implementan tres funciones de activación en sus propias clases, estas son [``Step``, ``Sigmoid`` y ``Tanh``](./Perceptrons/Activation.py). En estas se tienen métodos para aplicar la función, calcular la derivada de la función para un input y calcular la derivada función para un output de la misma. Estos métodos son _apply_, _derivative_ y _derivate_output_ respectivamente.


### Normalización, 1-hot encoding y partición

La normalización y la clasificación en 1-hot encoding se realiza con funciones implementadas en el archivo [utils.py](./Perceptrons/utils.py).

Para llevar a cabo la partición del dataset en conjuntos de entrenamiento y prueba, se decidió utilizar la librería [scikit-learn](https://scikit-learn.org/stable/).

### Apreciaciones

Entre las dificultades encontradas durante la implementación está el reconocimiento de los valores y/o variables relevantes que modificar y retornar (o no) en métodos como _feed_ y _train_. También hubo dificultades cuando se intentó optimizar la cantidad de variables de instancia de las distintas clases utilizadas.

En cuanto a la eficiencia de la red utilizada, se cree que trabaja en tiempos más lentos que una red implementada utilizando multiplicación de matrices, esto debido a que, entre otros, computa ejemplo por ejemplo, sin optimizar tiempos por batch de datos.

# Análisis

El análisis se realiza con datos del [_seeds dataset_](https://archive.ics.uci.edu/ml/datasets/seeds). Los datos se separan en atributos y clasificación; para los atributos se normalizan los datos y para la clasificación se codifican los valores. Posterior a esto se particionan los datos. La implementación de esto se encuentra en el archivo [seeds_training.py](./Perceptrons/seeds_training.py).


Se realizan tres 'tipos' de 'experimentos'. El primero es variar las funciones de activación y cantidad de neuronas por capa con dos capas escondidas. El segundo es variar la cantidad de neuronas por capa para redes con cuatro capas escondidas. El tercero es variar la tasa de aprendizaje para redes con dos capas escondidas y con _tanh_ como función de activación.

A continuación se muestran algunos resultados, viendo curvas de aprendizaje, error de la red en las distintas épocas de entrenamiento y la matriz de confusión luego de todas las épocas.

Todos los gráficos generados y descripción de dimensiones utilizadas se encuentran en el directorio [analisis_plot](./Perceptrons/analisis_plots).

## Sigmoid y dos capas escondidas
Estos son los resultados obtenidos al entrenar una red con todas sus neuronas con sigmoid como función de activación, 0.6 de tasa de aprendizaje y 2 capas escondidas.
### Curvas de aprendizaje
La curva de aprendizaje obtenida es similar a la esperada, con un aumento prácticamente sostenido, y luego de cierta cantidad de épocas se "estanca" en un porcentaje de aciertos razonables.

En casi todos los experimentos es así, menos al tener pocas (1 o 2) neuronas por capa, en la cual nunca se supera el 80% de aciertos.

![accuracy 2layers 1exp](./Perceptrons/analisis_plots/twohl/acc_1.png)

### Error de la red
El error obtenido muestra una curva que disminuye sus valores de forma suave, lo cual es lo esperado. El valor del error en la última época computada es muy cercano a 0.

![error 2layers 1exp](./Perceptrons/analisis_plots/twohl/mse_1.png)
### Matriz de confusión
La red logra clasificar la mayoría de los ejemplos de prueba correctamente, fallando solo en 5. Otras redes probadas en esta sección tienen una cantidad de fallos similares, menos las red con pocas neuronas por capa, que falla en 13 ejemplos.

![cm 2layers 1exp](./Perceptrons/analisis_plots/twohl/cm_1.png)


## Sigmoid y cuatro capas escondidas
Para estos experimentos se quiso observar como se comportaba una red al aumentar el numero de capas escondidas.

Si bien en general se ven más variaciones en los aciertos y errores entre épocas que en el caso de tener dos capas escondidas, la matriz de confusión final aquí mostrada no es tan diferente a lo esperado (diagonal diferenciable). El comportamiento es el esperado (aumento de aciertos, disminucion de error), la diferencia es que entre épocas se tiene una curva no suave.

Los gráficos aquí mostrados son resultados de la primera red neuronal con cuatro capas escondidas. Se debe notar que para la segunda red (con pocas neuronas por capas), se tiene un resultado con mucho error y una línea horizontal como curva de aprendizaje(no aprende nada). En este mismo experimento, al aumentar la cantidad de neuronas por capas se obtienen curvas más parecidas a las obtenidas al utilizar dos capas escondidas.

### Curvas de aprendizaje
Se observa que la red mejora su acierto, pues a partir de la época 25 está siempre sobre 0.6 de acierto. Sin embargo, el porcentaje presenta variaciones en un rango de 0.25. estas variaciones pueden deberse a que se tienen más capas escondidas que las óptimas para trabajar con el dataset utilizado.

![accuracy 4layers 1exp](./Perceptrons/analisis_plots/fourhl/acc_1.png)

### Error de la red
El error de la red va disminuyendo, pero es solo después de la época 30 que pareciera quedarse en los valores más bajos que puede alcanzar. Esto se condice con lo observado en el gráfico de aciertos.

![error 4layers 1exp](./Perceptrons/analisis_plots/fourhl/mse_1.png)
### Matriz de confusión
Se observa que la matriz de confusión de la red presenta una buena cantidad de aciertos, pero sigue fallando en algunos casos (lo que es razonable).

![cm 4layers 1exp](./Perceptrons/analisis_plots/fourhl/cm_1.png)


## Tanh y diferentes tasas de aprendizaje
Para estos experimentos se utilizan redes neuronales con dos capas escondidas, en donde todas las funciones de activación corresponden a tanh.

Estos gráficos se generaron debido a que la variación en el error y acierto -para una red similar a la de los primeros resultados presentados- era significativa, por lo que se quiso ver si fue un error de implementacion o si era por la naturaleza de la función.

### Curvas de aprendizaje
Los aciertos obtenidos para ambas redes aquí presentadas tienen el aumento esperado. Sin embargo, la red con un _learning rate_ mayor modifica mucho sus valores entre épocas, por lo que el porcentaje de aciertos no pareciera converger. En la red con neuronas con _learning rate_ igual a 0.2, no se tiene una curva tan suave como en la sección con dos capas y sigmoid, pero tiene menores variaciones que la otra red presentada en esta sección.


![error tanh 1exp](./Perceptrons/analisis_plots/tanh/acc_1.png "Learning rate 0.6")
![error tanh 3exp](./Perceptrons/analisis_plots/tanh/acc_3.png "Learning rate 0.2")


### Error de la red
Para el primer ejemplo, se tiene que el error va disminuyendo, pero tiene unas alzas y variaciones notorias entre épocas. En cuanto al segundo grafico, se tiene que el error disminuye de forma más suave, y que si bien tiene alzas, estas son de una amplitud menor.

A partir de estos resultados, se observa que tanh trabaja mejor con una tasa de aprendizaje más pequeña.

![error tanh 1exp](./Perceptrons/analisis_plots/tanh/mse_1.png "Learning rate 0.6")
![error tanh 3exp](./Perceptrons/analisis_plots/tanh/mse_3.png "Learning rate 0.2")

### Matriz de confusion
Si bien todos los experimentos pueden variar, aquí se presentan resultados similares a la mayoría de aquellos obtenidos.

Para la red que tiene una tasa de aprendizaje mayor (0.6), al final de 50 épocas realiza un muy mal trabajo clasificando los ejemplos, sobre todo al predecir la clase **2**. En cambio, para una red con tasa de aprendizaje 0.2, se tiene que puede clasificar en la clase **2**, teniendo un mayor porcentaje de aciertos.

![cm tanh 1exp](./Perceptrons/analisis_plots/tanh/cm_1.png "Learning rate 0.6")
![cm tanh 3exp](./Perceptrons/analisis_plots/tanh/cm_3.png "Learning rate 0.2")



# Instaleichon y usaje
Para correr el código se debe activar un entorno virtual de Python. 
Ver documentación de Python para [creacion de entornos virtuales.](http://docs.python.org.ar/tutorial/3/venv.html#creando-entornos-virtuales)

La instalación de paquetes necesarios para utilizar este código puede realizarse con pip y ejecutando la siguiente línea.
```bash
pip3 install -r requirements.txt
```

Para generar los gráficos y output del entrenamiento de una red con los datos del [dataset _seeds_](https://archive.ics.uci.edu/ml/datasets/seeds), se debe correr el programa seeds_training.py con la siguiente línea.
```bash
python3 Perceptrons/seeds_training.py
```