import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report

# Cargar los datos
ventas = pd.read_csv('data/ventas_pizzas.csv')
materia_prima = pd.read_csv('data/materia_prima.csv')

# Convertir la columna 'Fecha' a tipo datetime
ventas['Fecha'] = pd.to_datetime(ventas['Fecha'])

# Crear nuevas columnas para el análisis
ventas['Mes'] = ventas['Fecha'].dt.month
ventas['Año'] = ventas['Fecha'].dt.year

# Agrupar datos por Pizza y calcular las ventas totales
ventas_agg = ventas.groupby('Pizza').agg({'Cantidad': 'sum', 'Total Venta': 'sum'}).reset_index()

# Modelo de Regresión

# Variables independientes (X) y dependientes (y)
X = ventas[['Cantidad', 'Precio Unitario']]
y = ventas['Total Venta']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear el modelo de regresión
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = reg_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Modelo de Clasificación

# Crear una columna 'Clase' para la clasificación
ventas_agg['Clase'] = ['Más Vendida' if x >= 20 else 'Menos Vendida' for x in ventas_agg['Cantidad']]

# Variables independientes (X) y dependientes (y)
X_clf = ventas_agg[['Cantidad']]
y_clf = ventas_agg['Clase']

# Convertir la variable dependiente en valores numéricos
y_clf = y_clf.map({'Más Vendida': 1, 'Menos Vendida': 0})

# Dividir los datos en entrenamiento y prueba
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=0)

# Crear el modelo de clasificación
clf_model = RandomForestClassifier(n_estimators=100, random_state=0)
clf_model.fit(X_train_clf, y_train_clf)

# Hacer predicciones
y_pred_clf = clf_model.predict(X_test_clf)

# Evaluar el modelo
print(classification_report(y_test_clf, y_pred_clf))
