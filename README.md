# Schistosoma_Mansoni_models

This page shows python code and models trained from the paper "Machine learning models for predicting inhibitors for Schistosoma Mansoni Thioredoxin Peroxidase (Prx2)" on biochemical data of compounds screened against Schistosoma mansoni

Combining files, pre-processing, data splitting, training, and loading and testing are all included in the code on this section. Create a directory for the filepaths in the.py files for the tasks, to reiterate the code.

Follow these key steps to recreate the experiment:

1. Download the data (active and inactive separately) from PubChem in the SDF formats.
2. Feed the two files to the descriptor calculator (preferably Enalos Mold2 node on KNIME) to obtain two .csv files with calculated descriptors.
3. Feed the Active data into the SMOTE node (found on KNIME) where the synthetic samples are generated.
4. To merge the two.csv files into one, execute the Combining files.py code.
5. On the produced file, run Pre-processing.py.
6. On the pre-processed data, run Data splitting.py.
Feed the training, testing and validation dataset into the respective generated models by running; RF.py, Support_vector_machine.py, 
