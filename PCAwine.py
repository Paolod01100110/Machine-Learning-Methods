import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#  carica il dataset Wine
wine = load_wine()
X, y = wine.data, wine.target #caratteristiche numeriche ed etichette

# esploro il dataset
print("Numero di campioni per classe:", np.bincount(y)) #conta il numero di elementi per ciascuna classe
print("Statistiche di base delle feature:")
print(np.mean(X, axis=0))  # media delle feature

# Visualizzazione distribuzione delle classi
plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette="viridis") #grafico a barre
plt.xlabel("Classe")
plt.ylabel("Numero di campioni")
plt.title("Distribuzione delle classi nel dataset Wine")
plt.show()

# riduco la dimensionalità con PCA
scaler = StandardScaler() #standardizza i dati
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2) #trasformo le feature originali in 2 componenti principali
X_pca = pca.fit_transform(X_scaled)

# Visualizzazione PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis") #grafico a due dimensioni
plt.xlabel("Prima componente principale")
plt.ylabel("Seconda componente principale")
plt.title("Visualizzazione dati dopo PCA")
plt.show()

# suddivido il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# applico un algoritmo di classificazione (Random Forest)
clf = RandomForestClassifier(random_state=42) #insieme di alberi decisionali
clf.fit(X_train, y_train)

# 6. Valuta la performance del modello
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) #previsioni corrette
precision = precision_score(y_test, y_pred, average="weighted") #percentuale di predizioni positive corrette
recall = recall_score(y_test, y_pred, average="weighted") #percentuale di veri positivi identificati correttamente
f1 = f1_score(y_test, y_pred, average="weighted") #media armonica tra precision e recall

print("Metriche di valutazione del modello:")
print(f"Accuratezza: {accuracy:.4f}")
print(f"Precisione: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 7. Visualizza l'importanza delle feature
importances = clf.feature_importances_ #indica l'importanza delle feature per la classificazione
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(wine.feature_names)[indices], palette="viridis")
plt.xlabel("Importanza delle feature")
plt.ylabel("Feature")
plt.title("Importanza delle feature secondo il modello Random Forest")
plt.show()

# 8. Visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis", xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione")
plt.show()

# 9. Ottimizza l'algoritmo con GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1) #testa diverse combinazioni di iperparametri
grid_search.fit(X_train, y_train)

print("Migliori parametri trovati:", grid_search.best_params_)

# modello ottimizzato
best_model = grid_search.best_estimator_
y_pred_opt = best_model.predict(X_test)

# Nuova valutazione con il modello ottimizzato
accuracy_opt = accuracy_score(y_test, y_pred_opt)
print(f"Accuratezza dopo ottimizzazione: {accuracy_opt:.4f}")
