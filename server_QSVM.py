import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap


# Make feature map with ZZ 
from qiskit import transpile, QuantumCircuit
from qiskit_aer import Aer

backend = Aer.get_backend("qasm_simulator")
shots = 1024
dimention = 6
feature_map = ZZFeatureMap(dimention, reps=1)

# Define kernel
def evaluate_kernel(x_i, x_j):
    """
    Evaluates the dot product of two input vectors in a higher dimensional space using a quantum circuit.

    Parameters:
    -----------

    `x_i` : array-like, The first input vector of shape `(d,)`.
    `x_j` : array-like, The second input vector of shape `(d,)`.

    Returns:
    --------

    `float`: The dot product of `x_i` and `x_j` in the higher dimensional space.
    """
    circuit = QuantumCircuit(dimention)
    circuit.compose(feature_map.assign_parameters(x_i), inplace=True)
    circuit.compose(feature_map.assign_parameters(x_j).inverse(), inplace=True)
    circuit.measure_all()

    transpiled = transpile(circuit, backend)
    counts = backend.run(transpiled, shots=shots).result().get_counts()    
    return counts.get("0" * dimention, 0) / shots

# Qiskit imports
from qiskit.visualization import circuit_drawer
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.algorithms import QSVC

kernel = TrainableFidelityQuantumKernel(
    feature_map=feature_map
)

# Read file and run QSVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
import pickle

df = pd.read_csv('./uubar.csv')
AUC = {}
for kappa in [0, 0.3, 0.5, 0.7, 1]:
    AUC[kappa] = []

#for num in [10, 20, 50, 100, 200, 500]:
for num in [10, 20]:
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
        #print(X.shape)
        #print(y.shape)
        qsvm = SVC(kernel=kernel.evaluate)
        qsvm.fit(X_train, y_train)
        
        # Get the predicted probabilities or decision function scores
        if hasattr(qsvm, "predict_proba"):
            y_probs = qsvm.predict_proba(X_test)[:, 1]
        else:
            y_scores = qsvm.decision_function(X_test)
            # Normalize decision function scores to [0, 1] range for ROC AUC
            y_probs = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        # Compute the AUC score
        auc_score = round(roc_auc_score(y_test, y_probs), 4)
        print(f'AUC Score: {auc_score}')

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)

        # AUC append
        AUC[kappa].append(auc_score)        

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
        
        with open('./models/model_'+str(num)+'_'+str(kappa)+'.pkl','wb') as f:
            print(f, " is created")
            pickle.dump(qsvm,f)
print(AUC)            
'''
for kappa in [0, 0.3, 0.5, 0.7, 1]:    
    with open('./models/model_'+str(kappa)+'.pkl', 'rb') as f:
        clf = pickle.load(f)
        predict = clf.predict(X)
        print('model'+str(kappa)+':', clf.score(X, y))    
'''        