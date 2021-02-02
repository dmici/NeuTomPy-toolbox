============
Installation
============

NeuTomPy toolbox supports **Linux**, **Windows** and **Mac OS** 64-bit operating systems.

First of all, install a `Conda <https://www.anaconda.com/download/>`_  python environment with **Python 3.6** and then activate it:

.. code-block:: bash

    conda create -n ntp_env python=3.6 
    conda activate ntp_env

Install some dependencies:

.. code-block:: bash

    conda install -c simpleitk simpleitk
    conda install scikit-image
    conda install ipython numexpr astropy tifffile mkl_fft tqdm
    conda install -c astra-toolbox astra-toolbox
    pip install opencv-python read-roi
    pip install -U numpy

Finally, install NeuTomPy toolbox via ``pip``:

.. code-block:: bash

    pip install neutompy

NB: If a segmentation fault occurs when importing NeuTomPy, install PyQt5 via ``pip``:

.. code-block:: bash

    pip install PyQt5


Update
------

To update a NeuTomPy installation to the latest version run:

.. code-block:: bash

    pip install neutompy --upgrade
