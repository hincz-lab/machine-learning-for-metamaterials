Auxilary scripts needed to run the other scripts in this repository.

These modules should be available in the same versions for each of the other
folders with scripts where they are referenced. Alternatively you can load
the module directly from this folder for better version control.

Folder contents:
- python_environment.txt: printout of the python environment in which all
                scripts were ran, using the pip.list command. Python version
                3.7.0 was utilized in all scripts. Versions of all relevant
                packages should be matached to this file for best consistancy.
 
- BB_metals.py: implementation of the Brendel-Bormann model for optical
                constants of many metals, based on Rakic et.al. 1998.
                Load the module and run the "$ materials()" function to see
                a list of included materials.

                Usage: "$ nk_materials('foo',wavelengths)"

- dielectric_materials.py: module containing optical constants for several 
                common dielectrics, including several oscillator models and
                EMAs. Load the module and run the "$ materials()" function
                for a list of included interpolation materials and functions,
                along with proper usage for each method.

- dielectric_nk_data.h5: data file containing optical constants for several
                interpolation materials. All data is obtained from
                refractiveindexinfo.com for the dataset most similar to thin
                films in the visible range. File must be in the same directory
                with dielectric_materials.py for usage of the nk_materials() 
                function.

- LD_metals.py: implementation of the Lorentz-Drude model for optical
                constants of many metals, based on Rakic et.al. 1998. Load
                the model and run the "$ materials()" function to see a list
                of included materials.

                Usage: "$ nk_materials('foo',wavelengths)"

- TMM_numba.py: implementaion of the transfer matrix method for calculating
                the reflectance, transmittance, and ellipsometric spectra for
                stacked thin film metamaterials. Implemented with the numba
                compiler for faster runtime. Based on Chillwell et.al. 1984.
