Script to train the convolutional neural networks referenced in the work, and
fully trained CNNs from which the data is generated.

Each CNN has an independent training script by layer number and in a folder
dedicated to the specific inverse design problem solved. The python
environment and in particular Keras version must be updated to match the 
printout in /auxiliary_scripts/python_environment.txt for an accurate
reproduction of the claimed model results.

Trained models are included within the corresponding folders (foo/models/),
which are the models referenced in the work.

Folder Contents:
- ellipsometric2refl_trans/: CNNs mapping an ellipsometric spectral type to a
               reflectance/transmittance spectral type. Models contained in 
               ellipsometric2refl_trans/models/.

- ellipsometric2structure/: CNNs mapping an ellipsometric spectral type to a
               materials structure (materials and thicknesses). Models 
               contained in ellipsometric2structure/models/.

- refl_trans2ellipsometric/: CNNs mapping a reflectance/transmittance
               spectral type to an ellipsometric spectral type. Models
               contained in refl_trans2ellipsometric/models/.

- refl_trans2structure/: CNNs mapping a reflectance/transmittance spectral
               type to a materials structure (materials and thicknesses).
               Models contained in refl_trans2structure/models/.

- model_tests/: Contains ipynb notebooks to test the pretrained networks.
               Generates results for the loss, model metrics, spectral 
               RMSE and solution time. The notebook should be run on a
               system with identical resources to the comparison methods
               for an accurate timing comparison.
 
