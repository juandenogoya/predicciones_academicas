# predicciones_academicas
Predicciones sobre Dropout, Enrolled y Graduate

Data Set obtenido de Kaggle.
Contiene datos de estudiantes y diferentes variables independientes, con una variable objetivo "Target", de tipo Objet (Dropout, Enrolled, Graduate).

En el archivo "prediccion_abandono_finalizacion_full", se analizaron los diferentes caracteristicas del DS, trabajo con datos Null, Vacios.
Para luego aplicar tecnicas de Normalización, Split y finalizar con modelos de Clasificación.


En el archivo "prediccion_con_feature_engineering", similar al anterior en cuanto a trabajar con Nulos y Vacios, luego aplicaron tecnicas de feature engineering, aplicando librerias sklearn Polinomicas, creando variables de 2do grado.
Se quitaron variables, se aplicaron relaciones entre variables utilizando metodos de chi-cuadrado y Valor-p, transformaciones de la variable dependiente utilizanco "Label Encoder", normalizando y luego aplicar los modelos de clasificación Random Fores y XGBoost.


