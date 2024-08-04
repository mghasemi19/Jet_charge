import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit

algorithm_globals.random_seed = 42

if True:
    df = pd.read_csv('./uubar.csv')

# create empty array for callback to store evaluations of the objective function
objective_func_vals = []

# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration (Loss)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

from sklearn.model_selection import train_test_split

num_inputs = 6
Acc = {}
for kappa in [0, 0.3, 0.5, 0.7, 1]:
    Acc[kappa] = []

for num in [1000, 1500, 2000, 2500, 3000]:
#for num in [100, 202, 500, 1000, 2000, 5000]:
#for num in [200]:
    df_up = df[df['jet_type'] == 1].sample(n=num, random_state=42)
    df_antiup = df[df['jet_type'] == 0].sample(n=num, random_state=42)
    final_df = pd.concat([df_up, df_antiup], ignore_index=True)
    cols = list(final_df.columns)

    for kappa in [0, 0.3, 0.5, 0.7, 1]:    
    #for kappa in [0]:    
        columns = [col for col in cols if col.startswith('Q_') and col.endswith('_'+str(kappa))]
        #print(columns)
        X = final_df[columns].values
        y = final_df['jet_type'].values.reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # construct QNN with the QNNCircuit's default ZZFeatureMap feature map and RealAmplitudes ansatz.
        qc = QNNCircuit(num_qubits=num_inputs)
        
        # QNN with Estimator primitive
        estimator_qnn = EstimatorQNN(circuit=qc)
        # QNN maps inputs to [-1, +1]
        estimator_qnn.forward(X_train[0, :], algorithm_globals.random.random(estimator_qnn.num_weights))      # input like the first elemnt and weight as random array

        # construct neural network classifier
        estimator_classifier = NeuralNetworkClassifier(
            estimator_qnn, optimizer=COBYLA(maxiter=60)
        )
        # fit classifier to data
        estimator_classifier.fit(X_train, y_train)
        # score classifier   
        score = estimator_classifier.score(X_test, y_test)
        print(score)
        Acc[kappa].append(score)
              
print(Acc)
