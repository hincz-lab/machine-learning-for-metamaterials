# Machine Learning For Metamaterials

Code associated with the work "General Inverse Design of Thin-Film Metamaterials With Convolutional Neural Networks". 

## Description:

Optical applications of thin-film metamaterials generally involve the inverse design of metamaterial structure from a target spectral response. For systems with many choices of material structure, the large input parameter space can be an impdement to traditional optimization methods. We have applied convolutional netural networks (CNNs) to solve the inverse design problem. Furthermore we show how CNNs can solve all faects of the inverse design problem by explicating all relationships between material structure, reflectance / transmittance spectra and ellipsometrtic specta. We then show that CNNs can outperform traditional inverse design techniques for the parameter space considered here.

This repository contains all code necessary to reporoduce our results for systems of up to 5 layers with 5 distinct choices for materials, including the generation of thin-film metamaterial data, training of machine learning models, evaluation of the models, and comparison with other inverse design methods.

Working script to evaluate the CNN models feautred in the main work, along with test examples can be found in the associated Google Colabs environment. This includes evalaution script and the pretrained network models. This can be accessed at the following link:
https://colab.research.google.com/drive/1zDr5YTg8UMT10jA3jcxv8cb_XGu0qub1?usp=sharing. 

The trained models referenced in this work are hosted with OSF (Open Science Framework) due to size constraints. These models, along with some sample datasets, can be found at:
https://osf.io/65tyh/.

__Trained models are not included in this repository due to size restrictions. The pretrained models evaluated in the main work are available in the associated OSF repository, listed above. Alternatively, models can be obatined by contacting the corresponding authors.__


## File types included in this repository:
- py 
- ipynb
- txt
- h5


## Necessary Software to run all code in this repository:
- py   : Python 3.7.0 (recommended to use anaconda)
- ipynb: Jupyter Notebook
- h5 reader, recommended h5py
- See /auxilary_scripts/python_environment.txt for a snapshot of the intstalled python packages in the environment (pip.list readout) in which the results were produced. It is recommended to run all scripts in this repository with the indicated package versions for compatability.
