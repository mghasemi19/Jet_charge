# Jet_tagging

<div style="text-align: justify;">
Jet charge computation using classical and quantum machine learning is an innovative approach in high-energy particle physics, specifically in the analysis of data from particle colliders like the Large Hadron Collider (LHC). The jet charge is a weighted sum of the electric charges of the particles within a jet, typically weighted by their transverse momenta (example plot below). This quantity provides important insights into the properties of the originating quark or gluon and helps in distinguishing between different types of jets, such as those originating from up-type versus down-type quarks. By leveraging machine learning algorithms, we can enhance the precision and efficiency of jet charge determination. These algorithms can learn complex patterns and correlations within the high-dimensional data, outperforming traditional methods. Techniques such as deep neural networks, gradient boosting, and other supervised learning approaches are trained on simulated datasets to accurately predict the jet charge. This advancement not only improves the accuracy of jet classification but also aids in probing the fundamental interactions and properties of particles, potentially uncovering new physics beyond the Standard Model.
</div>

<p align="center">
<img width="400" alt="Screen Shot 2024-08-06 at 4 38 10 PM" src="https://github.com/user-attachments/assets/21057e19-f044-428d-959b-d4fc74660e07">
</p>

Here is the sequence how to run the code to make the ntuples generated with MadGraph and PYTHIA and run the quantum and classical machine learning models:
- Inside the [uubar](./uubar.ipynb) notebook, we use `write_mg_cards` function to generate `p p > u g` adn `p p > u~ g` processes in MadGraph for arbitrary number of events.
- `run_pythia_get_images` function can be used to run the PYTHIA and produce the ntuple with necessary variables such as jet's 4-vector, charge weighted with transverse momentum and mass, etc.
- All the selection for objects can be handeled in `run_pythia_get_images` function.
- Several calssical machine learning models are then applied to classify up and anti-up jets using input variables. For DNN and SVM models, hyperparameters are optimized.
- `make_image_leading_jet` and `make_image_event` functions extract information (4-vector) from the jet's tracks. The output numpy histogram2d are feeded to CNN model for training.
- To run Graph NN and extract the best nodes, [server_GNN](./server_GNN.py) can be run.
- Support vector machine with quantum kernel is in this file [QSVM](./server_QSVM.py). 
- Variational quantum classifier with quantum kernel is in this file [QSVM](./server_VQC.py). 
- Quantum neural network with quantum kernel is in this file [QSVM](./server_QNN.py). 
  
