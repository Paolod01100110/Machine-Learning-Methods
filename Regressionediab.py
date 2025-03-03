# Import delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# carica il dataset Diabetes
diabetes = load_diabetes()
X = diabetes.data  # le dieci misurazioni cliniche, cioè le variabili indipendenti
y = diabetes.target  # progressione della malattia del diabete, la variabile dipendente y

# esplora i dati
print("Feature names:", diabetes.feature_names) #stampa i nomi delle 10 caratteristiche cliniche
print("Shape dei dati X:", X.shape) #dimensioni
print("Shape dei dati y:", y.shape)

# crea un DataFrame per una migliore comprensione
df = pd.DataFrame(X, columns=diabetes.feature_names) #conversione in dataframe
df['Target'] = y #aggiunge la colonna target
print("\nAnteprima del dataset:")
print(df.head())

# suddivido il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# crea e addestra il modello di regressione lineare
model = LinearRegression()
model.fit(X_train, y_train)

# effettua le predizioni sul set di test
y_pred = model.predict(X_test)

# valuta le prestazioni del modello
mse = mean_squared_error(y_test, y_pred)  # Errore quadratico medio
r2 = r2_score(y_test, y_pred)  # Coefficiente di determinazione R²

print("\nValutazione del modello:")
print(f"Errore Quadratico Medio (MSE): {mse:.2f}")
print(f"Coefficiente di Determinazione (R²): {r2:.2f}")

# analisi dei risultati: grafico dei valori predetti vs valori reali
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dashed') #definisce la linea rossa con i punti estremi
plt.xlabel("Valori Reali")
plt.ylabel("Valori Predetti")
plt.title("Regressione Lineare - Valori Reali vs Predetti")
plt.show()


y_pred_rounded = np.round(y_pred)  # Arrotondiamo le predizioni ai numeri interi
accuracy_classification = np.mean(y_pred_rounded == y_test)  # calcola l'accuracy


metrics = ['MSE', 'Accuracy']
values = [mse, accuracy_classification]

plt.figure(figsize=(8,5))
plt.bar(metrics, values, color=['blue', 'orange'])
plt.ylabel("Valore")
plt.title("Confronto tra MSE e Accuracy") 
# Mostra i valori sopra le barre
for i, v in enumerate(values):
    plt.text(i, v + (max(values) * 0.05), f"{v:.2f}", ha='center', fontsize=12)

plt.show()

mse, accuracy_classification   