import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap


# Make feature map with ZZ 
from qiskit import transpile, QuantumCircuit
from qiskit_aer import Aer

backend = Aer.get_backend("qasm_simulator")
shots = 1024
dimention = 10
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


df = pd.read_csv('./uubar.csv')
df_up = df[df['jet_type'] == 1].sample(n=500, random_state=42)
df_antiup = df[df['jet_type'] == 0].sample(n=500, random_state=42)
final_df = pd.concat([df_up, df_antiup], ignore_index=True)
cols = list(final_df.columns)
cols = [col for col in cols if col.startswith('Q_1')]
print(cols)
X = final_df[cols].values
y = final_df['jet_type'].values.reshape(-1)
print(type(X))
print(X.shape)
print(y.shape)

from sklearn.svm import SVC
qsvm = SVC(kernel=kernel.evaluate) 
qsvm.fit(X, y)
predicted = qsvm.predict(X)
print(qsvm.score(X, y))