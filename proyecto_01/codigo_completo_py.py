# Prediciones de ABANDONO y FINALIZACION de Estudios


# LIBRERIAS del PROYECTO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import operator

#Transformaciones:
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Normalizar:
from sklearn.preprocessing import MinMaxScaler

#MODELOS
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#MEDICIONES:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.metrics import confusion_matrix


ruta =  'C:/Users/juand/OneDrive/Escritorio/PORTFOLIO - DS 2024/KAGGLE/proyecto_01/data_estudiantes.csv'
data= pd.read_csv(ruta, delimiter=';')
data.head(5)


# INICIO ANALISIS
data.columns
data.isnull().sum()

data.count()
data.info()

# Modificar Nombre de Variables
old_names = {
    "Mother's qualification": "mom_qualification",
    "Father's qualification": "dad_qualification",
    "Mother's occupation": "mom_occupation",
    "Father's occupation": "dad_occupation",
    "Daytime/evening attendance	": "daytime_evening",
    'Tuition fees up to date': 'matricula_al_dia',
    'GDP': 'pbi',
    'Curricular units 1st sem (grade)': 'curricular_primer_semestre',
    'Curricular units 2nd sem (grade)': 'curricular_segundo_semestre'

}

new_data = data.rename(columns=old_names)
new_data.columns

new_data.describe().T
new_data
new_data.drop(['curricular_primer_semestre','curricular_segundo_semestre' ], axis=1, inplace=True)
new_data

# AGRUPACIONES VARIAS 
new_data.columns

# Primeros INSIGHT
agrup_target = new_data.groupby("Target").size()
agrup_target

# AGRUPADO Target y Estado Civil
# * 1: Soltero/a
# * 2: Casado/a
# * 3: Viudo/a
# * 4: Divorciado/a
# * 5: Separado/a
# * 6: Otra situación (por ejemplo, unión libre, poligamia)

grupo_estado_civil = new_data.groupby(['Target', 'Marital status']).size().unstack()
grupo_estado_civil

plt.figure(figsize=(10, 6))
grupo_estado_civil.T.plot(kind='line', color=['blue', 'green', 'red'])
plt.title('Estado civil vs. Resultado')
plt.xlabel('Estado civil')
plt.ylabel('Cantidad')
plt.show()

# AGRUPADO Target y Genero
# * 0: Femenino
# * 1: Masculino


grupo_genero = new_data.groupby(['Target', 'Gender']).size().unstack()
grupo_genero

plt.figure(figsize=(10, 6))
grupo_genero.T.plot(kind='line', color=['blue', 'green', 'red'])
plt.xticks([0, 1]) 
plt.title ('Genero Vs. Resultado')
plt.xlabel('Genero')
plt.ylabel('Cantidad')
plt.show()


# AGRUPADO Target y Nacionalidad

grupo_nacionalidad = new_data.groupby(['Target', 'Nacionality']).size().unstack()
grupo_nacionalidad = grupo_nacionalidad.fillna(0)
grupo_nacionalidad


# AGRUPADO Ratio de Desempleo, Inflación y PBI

# Desempleo

grupo_desempleo = new_data.groupby(['Unemployment rate', 'Target']).size().unstack()
grupo_desempleo = grupo_desempleo.fillna(0)
grupo_desempleo

plt.figure(figsize= (6,3))
grupo_desempleo.plot(kind='line', color= ['blue', 'red', 'green'])
plt.title ('Desempleo Vs. Resultado')
plt.xlabel('Desempleo')
plt.ylabel('Cantidad')
plt.show()





# Inflación:

grupo_inflacion = new_data.groupby(['Inflation rate','Target']).size().unstack()
grupo_inflacion = grupo_inflacion.fillna(0)
grupo_inflacion

plt.figure(figsize=(6,3))
grupo_inflacion.plot(kind='line', color=['blue', 'red', 'green'])
plt.title('Inflación Vs. Resultado')
plt.xlabel('Ratio Inflación')
plt.ylabel('Cantidad')
plt.show()


# PBI

grupo_pbi = new_data.groupby(['pbi','Target']).size().unstack()
grupo_pbi = grupo_pbi.fillna(0)
grupo_pbi

plt.Figure(figsize=(6,3))
grupo_pbi.plot(kind='line', color=['blue', 'red', 'green'])
plt.title('PBI vs Resultado')
plt.xlabel('PBI')
plt.ylabel('Cantidad')
plt.show()


# AGRUPADO Comportamiento Pago Matricula

grupo_pago_matricula = new_data.groupby(['matricula_al_dia','Target']).size().unstack()
grupo_pago_matricula = grupo_pago_matricula.fillna(0)
grupo_pago_matricula

plt.figure(figsize=(6,3))
grupo_pago_matricula.plot(kind='line', color=['blue', 'red', 'green'])
plt.xticks([0,1])
plt.title('Pagos Matricula Vs Resultados')
plt.xlabel('Pagos Al dia')
plt.ylabel('Cantidad')
plt.show()


# AGRUPADO Edad de Inscripcion

grupo_edad_insc = new_data.groupby([ 'Age at enrollment','Target']).size().unstack()
grupo_edad_insc = grupo_edad_insc.fillna(0)
grupo_edad_insc

plt.figure(figsize=(6,3))
grupo_edad_insc.plot(kind='line')
plt.title ('Edad Inscripción Vs Resultados')
plt.xlabel('Edad Inscripción')
plt.ylabel('Cantidad')
plt.show()

# AGRUPADO Hora Cursado

grupo_hora_cursado = new_data.groupby(['daytime_evening', 'Target']).size().unstack()
grupo_hora_cursado = grupo_hora_cursado.fillna(0)
grupo_hora_cursado

plt.figure(figsize=(6,3))
grupo_hora_cursado.plot(kind='line', color=['blue', 'red', 'green'])
plt.xticks([0,1])
plt.xlabel('Hora Cursado')
plt.ylabel('Cantidad')
plt.show()

# AGRUPADO Educacion de Padres:
# * Nivel 1: Educacion Inicial
# * Nivel 50: Educación Profesional Avanzado

grupo_qualy_mom = new_data.groupby(['mom_qualification', 'Target']).size().unstack()
grupo_qualy_mom = grupo_qualy_mom.fillna(0)
grupo_qualy_mom


grupo_qualy_dad = new_data.groupby(['dad_qualification','Target' ]).size().unstack()

grupo_qualy_dad = grupo_qualy_dad.fillna(0)
grupo_qualy_dad

# Grafico Qualy Padres

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
grupo_qualy_mom.plot(kind='line', color=['blue', 'red', 'green'], ax=axes[0])
axes[0].set_xlabel('Qualy Madre')
axes[0].set_ylabel('Cantidad')

grupo_qualy_dad.plot(kind='line', color=['blue', 'red', 'green'], ax=axes[1])
axes[1].set_xlabel('Qualy Padre')
axes[1].set_ylabel('Cantidad')

plt.suptitle('Comparación de Qualy Madre vs. Qualy Padre')
plt.tight_layout()
plt.show()


# Grafico Indicadores Economicos

fig, axes = plt.subplots(1, 3, figsize=(15, 3))
grupo_desempleo.plot(kind='line', color=['blue', 'red', 'green'], ax= axes[0])
axes[0].set_xlabel('Ratio Desempleo')
axes[0].set_ylabel('Cantidad')

grupo_inflacion.plot(kind='line', color=['blue', 'red', 'green'], ax= axes[1])
axes[1].set_xlabel('Ratio Inflación')
axes[1].set_ylabel('Cantidad')

grupo_pbi.plot(kind='line', color=['blue', 'red', 'green'], ax= axes[2])
axes[2].set_xlabel('PBI')
axes[2].set_ylabel('Cantidad')

plt.suptitle('Ratios Economicos Vs Resultados')
plt.tight_layout()
plt.show()


# Hora Cursado y Edad de Inscrpcion

fig, axes = plt.subplots(1,2, figsize= (12,3))

grupo_edad_insc.plot(kind='line', color=['blue', 'red', 'green'], ax=axes[0])
axes[0].set_xlabel('Edad Inscripción')
axes[0].set_ylabel('Cantidad')

grupo_hora_cursado.plot(kind='line', color= ['blue', 'red','green'], ax= axes[1])
axes[1].set_xlabel('Hora Cursado')
plt.xticks([0,1])
axes[1].set_ylabel('Cantidad')

plt.suptitle('Edad y Hora de cursado')
plt.tight_layout()
plt.show()



# INICIO TRANSFORMACION y SPLIT de DATOS
# TRANFORMACION Variable Dependiente (Target)

label_encoder = LabelEncoder()
new_data ['Target'] = label_encoder.fit_transform(new_data['Target'])
new_data['Target'].unique()

# SPLIT de DATOS:
X_train, X_test, y_train, y_test = train_test_split(new_data.drop('Target', axis=1), new_data['Target'], test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# NORMALIZACION de Datos:
scaler_train = MinMaxScaler()
scaler_test = MinMaxScaler()

X_train_normalized = scaler_train.fit_transform(X_train)
X_test_normalized = scaler_test.fit_transform(X_test)

# SELECCION del MODELO a ENTRENAR:


# RANDOM FOREST
modelo_random_forest = RandomForestClassifier(n_estimators=100)
new_data_RFC = modelo_random_forest.fit(X_train_normalized, y_train)

predicciones = modelo_random_forest.predict(X_test_normalized)

# CALCULO METRICAS:
precision = precision_score(y_test, predicciones, average='weighted')
recall = recall_score(y_test, predicciones, average='weighted')
f1 = f1_score(y_test, predicciones, average='weighted')

print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# MATRIZ de CONFUSION:
print(confusion_matrix(y_test, predicciones))

mat_confusion_rfc = np.array([[223,15,46], [34,49,66], [18,14,420]])
mat_confusion_rfc_norm = mat_confusion_rfc.astype('float') / mat_confusion_rfc.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
im = ax.imshow(mat_confusion_rfc_norm, cmap='Reds')

etiqueta_classes_rfc = ['Dropout', 'Enrolled', 'Graduate']

ax.set_xticks(np.arange(len(etiqueta_classes_rfc)))
ax.set_yticks(np.arange(len(etiqueta_classes_rfc)))
ax.set_xticklabels(etiqueta_classes_rfc, rotation=45, ha='right')
ax.set_yticklabels(etiqueta_classes_rfc)
ax.set_title('Matriz de Confusión')
fig.colorbar(im, label='Proporción')

plt.tight_layout()
plt.show()


# XGBoost
model_XGB_GS = xgb.XGBClassifier()

parametros_GS = {'max_depth': [3,5,7], 'learning_rate': [0.1,0.01,0.001], 'n_estimators': [100,200,300]}
grid_search = GridSearchCV(estimator=model_XGB_GS, param_grid=parametros_GS, scoring='accuracy', cv=5)
grid_search.fit(X_train_normalized, y_train)

print(grid_search.best_params_)

# CALCULO METRICAS
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_normalized)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("F1:", f1)


# CROSS Validation
new_data_cross_val = RandomForestClassifier()
scores = cross_val_score(new_data_cross_val, X_train_normalized, y_train, cv=5)

# CALCULO METRICAS:
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# MATRIZ de CONFUSION:
print(confusion_matrix(y_test, y_pred))

matriz_confusion = np.array([[227,  20,  37], [ 33,  63,  53], [ 18,  19, 415]])
matriz_confusion_norm = matriz_confusion.astype('float') / matriz_confusion.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
im = ax.imshow(matriz_confusion_norm, cmap='Blues')

etiqueta_classes = ['Dropout', 'Enrolled', 'Graduate']

ax.set_xticks(np.arange(len(etiqueta_classes)))
ax.set_yticks(np.arange(len(etiqueta_classes)))
ax.set_xticklabels(etiqueta_classes, rotation=45, ha='right')
ax.set_yticklabels(etiqueta_classes)
ax.set_title('Matriz de Confusión')
fig.colorbar(im, label='Proporción')

plt.tight_layout()
plt.show()
