Scripts and data necessary to generate the full RMSE space for solutions to
the inverse design problem, from which the CNN can generate solutions.

The full RMSE space represents the solutions space for the inverse design
problem reference here, and can be fully explored for few-layer systems. This
allows one to test the performance of the optimization methods.

Folder Contents:
generate_mse-space_<N>-layer.py: script to generate the RMSE space for a
            given number of layers (<N> layers). This script will only
            generate RMSE results for a range of thicknesses, it is
            recommended to run multiple iterations of the same script on
            multiple nodes simultaneously to speed data generation.
            Modifications to accomplish this are included in the script. The
            script must also be modified to point at the intended spectral
            inputs. Warning: generates ~100gb data in txt files for 5-layer
            systems.

anti_reflection/: reflectance and transmittance spectra (Rp, Rs, Tp, Ts) for
            a typical anti-reflection filter. This data is drawn and not 
            physically generated. Consists of non-dispersive drawn spectra.

notch_filter/: reflectance and transmittance spectra (Rp, Rs, Tp, Ts) for a
            typical notch filter. This data is drawn from several gaussian
            and inverted gaussian peaks and not physically generated.

tanh_lpf/: reflectance and transmittance spectra (Rp, Rs, Tp, Ts) for a
            typical low pass filter with relatively sharp cutoff. This data
            is generated from several hyperbolic tangent functions and is not
            physically generated.

auxiliary files: See /auxiliary_scripts/README.txt for a discription of
            these files and their typical usage.
