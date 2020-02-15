# NeuTomPy toolbox
<img src="https://github.com/dmici/NeuTomPy-toolbox/blob/master/img/logo_neutompy.png" width="850">

**NeuTomPy toolbox** is a Python package for tomographic data processing and reconstruction.
Such toolbox includes pre-processing algorithms, artifacts removal and a wide range of iterative
reconstruction methods as well as the Filtered Back Projection algorithm.
The NeuTomPy toolbox was conceived primarily for Neutron Tomography and developed to support
the need of users and researchers to compare state-of-the-art reconstruction methods and choose the optimal data-processing workflow for their data.

# Features
* Readers and writers for TIFF and FITS files and stack of images
* Data normalization with dose correction, correction of the rotation axis tilt, ring-filters, outlier removals, beam-hardening correction
* A wide range of reconstruction algorithms powered by [ASTRA toolbox](https://www.astra-toolbox.com/): FBP, SIRT, SART, ART, CGLS, NN-FBP, MR-FBP
* Image quality assessment with several metrics

# Installation

NeuTomPy toolbox supports **Linux**, **Windows** and **Mac OS** 64-bit operating systems.

First of all, install a [conda](https://www.anaconda.com/download/) python environment with  **Python 3.5 or 3.6**.

It is required to install some dependencies, hence run the following inside a conda environment:
```
conda install -c simpleitk simpleitk
conda install -c astra-toolbox astra-toolbox
conda install -c conda-forge ipython numpy numexpr matplotlib astropy tifffile opencv scikit-image read-roi mkl_fft scipy six tqdm pywavelets
```

Then install NeuTomPy toolbox via `pip`:

```
pip install neutompy
```

NB: If a segmentation fault occurs when importing NeuTomPy, install PyQt5 via `pip`:

```
pip install PyQt5
```

# Update

To update a NeuTomPy installation to the latest version run:
```
pip install neutompy --upgrade
```

# Documentation
Complete documentation can be found on Read the Docs: <https://neutompy-toolbox.readthedocs.io>.

Tutorials and code examples of typical usage can be found in the folder [examples](https://github.com/dmici/NeuTomPy-toolbox/blob/master/examples).

A sample dataset for testing purpose can be found [here](https://mega.nz/#F!k0g32QiC!zbGZMuTES4WOzrxJEfPaSA). This dataset includes neutron radiographs of a phantom sample acquired at the IMAT beamline, ISIS neutron spallation source, UK.

# Reference
If you use the NeuTomPy toolbox for your research, please cite the following paper:

D. Micieli, T. Minniti, G. Gorini, “NeuTomPy toolbox, a Python package for tomographic data processing and reconstruction”, SoftwareX, Volume 9 (2019), pp. 260-264, https://doi.org/10.1016/j.softx.2019.01.005.


# License
The project is licensed under the [GPLv3](https://github.com/dmici/NeuTomPy-toolbox/blob/master/LICENSE) license.

# Contact
If you want to contact us for any reasons, please send an email to: neutompy@gmail.com
