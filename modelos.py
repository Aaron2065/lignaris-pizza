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
X = ventas[['Cantidad', 'Precio Unitario']]
y = ventas['Total Venta']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Modelo de Clasificación
ventas_agg['Clase'] = ['Más Vendida' if x >= 20 else 'Menos Vendida' for x in ventas_agg['Cantidad']]
X_clf = ventas_agg[['Cantidad']]
y_clf = ventas_agg['Clase']
y_clf = y_clf.map({'Más Vendida': 1, 'Menos Vendida': 0})
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=0)
clf_model = RandomForestClassifier(n_estimators=100, random_state=0)
clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf)
print(classification_report(y_test_clf, y_pred_clf))

# Análisis Adicional
# Total de ventas por mes y año
ventas_mensual = ventas.groupby(['Año', 'Mes']).agg({'Total Venta': 'sum'}).reset_index()

# Promedio de ventas por pizza
promedio_ventas_pizza = ventas.groupby('Pizza').agg({'Total Venta': 'mean'}).reset_index()
promedio_ventas_pizza.rename(columns={'Total Venta': 'Promedio Venta'}, inplace=True)

# Total de materia prima utilizada por pizza (simulando un cálculo)
ventas_materia_prima = ventas.groupby('Pizza').agg({'Cantidad': 'sum'}).reset_index()

# Guardar los análisis en diferentes archivos CSV
ventas_agg.to_csv('data/ventas_agg.csv', index=False)
ventas_mensual.to_csv('data/ventas_mensual.csv', index=False)
promedio_ventas_pizza.to_csv('data/promedio_ventas_pizza.csv', index=False)
ventas_materia_prima.to_csv('data/ventas_materia_prima.csv', index=False)

# Imprimir el contenido de los nuevos archivos CSV
print("\nVentas agregadas guardadas en 'data/ventas_agg.csv':")
print(ventas_agg.head())
print("\nVentas mensuales guardadas en 'data/ventas_mensual.csv':")
print(ventas_mensual.head())
print("\nPromedio de ventas por pizza guardadas en 'data/promedio_ventas_pizza.csv':")
print(promedio_ventas_pizza.head())
print("\nVentas de materia prima guardadas en 'data/ventas_materia_prima.csv':")
print(ventas_materia_prima.head())
