# Machine Learning For Metamaterials

Code associated with the work "General Inverse Design of Thin-Film Metamaterials With Convolutional Neural Networks". 

## Description:

Optical applications of thin-film metamaterials generally involve the inverse design of metamaterial structure from a target spectral response. For systems with many choices of material structure, the large input parameter space can be an impdement to traditional optimization methods. We have applied convolutional netural networks to solve the inverse design problem, and show how they can solve the inverse design problem. Furthermore we show that these methods substantially improve runtime over traditional optimization methods.

This repository contains all code necessary to reporoduce our results for systems of up to 5 layers with 5 distinct choices for materials, including the generation of thin-film metamaterial data, training of machine learning models, evaluation of the models, and comparison with other inverse design methods.

__Trained models are not included in this repository due to size restrictions. The pretrained models evaluated in the main work can be obatined by contacting the  corresponding authors.__


## File types included in this repository:
- py 
- ipynb
- txt
- h5


## Necessary Software to run all code in this repository:
- py   : Python 3.7.0 (recommended to use anaconda)
- ipynb: Jupyter Notebook
- See /auxilary_scripts/python_environment.txt for a snapshot of the intstalled python packages in the environment (pip.list readout) in which the results were produced. It is recommended to run all scripts in this repository with the indicated package versions for compatability.
