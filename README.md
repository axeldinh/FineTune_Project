# Transfer-learning through Fine-tuning:<br/> What Parameters to Freeze?

Here we provide the code used for the semester project: Transfer-learning through Fine-tuning: What Parameters to Freeze?

This project has been supervised by Martin Jaggi and Matteo Pagliardini from the Machine Learning and Optimization Laboratory, EPFL, Switzerland.

Libraries
=========

pytorch, transformers

Data
====

The data should be in the main folder as such:

MainFolder/data/dataset/set.tsv, with dataset in ['CoLA', 'RTE, 'SST-2', 'QNLI'] and set in ['dev', 'test'].
          
Otherwise, the PATH from the hyperparameters dict that can be found in the notebooks should be modified.

The datasets can be found on this link: https://gluebenchmark.com/tasks.

Training
========

The motebooks contain the training for 1 dataset. The seed defined here are the one used for the experiment, so that running the notebooks do not run everything again.
To run the experiment again, either delete the results in Results/Models/..., or change the seed.

The training were done using Adam on various learning rates, no other tricks were used.

Results
=======

In the Results folder, all the models trained for the report, as well as the figures and other objects can be found.

In models, the results are sorted per dataset.
