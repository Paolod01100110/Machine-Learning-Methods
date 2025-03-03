import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def generate_data():
    np.random.seed(42)
    data = {
        'Feature1': np.random.randn(100), #genera 100 valori casuali
        'Feature2': np.random.randn(100),
        'Target': np.random.choice([0, 1], size=100) #genera la colonna target con valori compresi tra 0 e 1
    }
    df = pd.DataFrame(data) #conversione in dataframe
    print("Dati generati con successo!\n")
    return df

def descriptive_analysis(df):
    print("Analisi descrittiva:\n") #min max stnd dev
    print(df.describe())
    print("\nDistribuzione delle classi:")
    print(df['Target'].value_counts()) #conta le occorrenze di ciascuna classe

def plot_data(df):
    plt.scatter(df['Feature1'], df['Feature2'], c=df['Target'], cmap='coolwarm', edgecolors='k') #grafico a dispersione
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.title('Distribuzione dei Dati')
    plt.colorbar(label='Target')
    plt.show()

def classification_results(df):
    X = df[['Feature1', 'Feature2']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #training set, test set
    #grid search con validazione incrociata per ottimizzare
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_ #estrae il miglior modello trovato
    
    y_pred = best_model.predict(X_test) #effettua le predizioni sul test set
    
    print("Classification Report:") #classificatio report
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred) #matrice di confusione
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice di Confusione')
    plt.show()

def main():
    df = None #all'inizio non ci sono dati
    while True:
        print("\nMenu di Selezione:")
        print("1. Generazione Dati")
        print("2. Analisi Descrittiva")
        print("3. Grafico dei Dati")
        print("4. Report di Classificazione con Matrice di Confusione")
        print("5. Esci")
        
        choice = input("Seleziona un'opzione: ")
        
        if choice == '1':
            df = generate_data() #crea un dataset
        elif choice == '2':
            if df is not None:
                descriptive_analysis(df) #analisi descrittiva 
            else:
                print("Genera prima i dati!")
        elif choice == '3':
            if df is not None:
                plot_data(df) #scatter plot
            else:
                print("Genera prima i dati!")
        elif choice == '4':
            if df is not None:
                classification_results(df) #genera un report di classificazione
            else:
                print("Genera prima i dati!")
        elif choice == '5':
            print("Uscita in corso...")
            break
        else:
            print("Scelta non valida, riprova.")

if __name__ == "__main__":
    main()
