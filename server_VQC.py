import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.algorithms import VQC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import pickle

# Read input file
df = pd.read_csv('./uubar.csv')

# number of features
num_features = 6

# Ansatz
feature_map = ZZFeatureMap(num_features, reps=1)
model = EfficientSU2(num_features, reps=1, entanglement="pairwise")

circuit = feature_map.compose(model)
circuit.measure_all()

# Read file and run VQC
df = pd.read_csv('./uubar.csv')
AUC = {}
Acc = {}
for kappa in [0, 0.3, 0.5, 0.7, 1]:
    AUC[kappa] = []
    Acc[kappa] = []

#for num in [1000, 1500, 2000]:
for num in [10, 20, 50, 100, 200, 500, 1000, 1500, 2000]:
    print("num:", num)
    df_up = df[df['jet_type'] == 1].sample(n=num, random_state=42)
    df_antiup = df[df['jet_type'] == 0].sample(n=num, random_state=42)
    final_df = pd.concat([df_up, df_antiup], ignore_index=True)
    cols = list(final_df.columns)

    for kappa in [0, 0.3, 0.5, 0.7, 1]:    
        columns = [col for col in cols if col.startswith('Q_') and col.endswith('_'+str(kappa))]
        #print(columns)
        X = final_df[columns].values
        y = final_df['jet_type'].values.reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                    
        vqc = VQC(num_features, feature_map, model)
        vqc.fit(X_train, y_train)  
        print(vqc.score(X_test, y_test))
        # Predict probabilities
        y_probs = vqc.predict(X_test)

        # Compute the AUC score
        auc_score = roc_auc_score(y_test, y_probs)
        print(f'AUC Score: {auc_score:.4f}')

        predict = vqc.predict(X_test)     
        acc = round(vqc.score(X_test, y_test), 4)    
        print(f'ACC: {acc}')

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)

        # AUC append
        AUC[kappa].append(auc_score) 
        Acc[kappa].append(acc)        

        # Plot the ROC curve
        if False:
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()
print(AUC)         
print(Acc)
