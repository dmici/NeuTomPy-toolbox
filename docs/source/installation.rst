============
Installation
============

NeuTomPy toolbox supports **Linux**, **Windows** and **Mac OS** 64-bit operating systems.

First of all, install a `Conda <https://www.anaconda.com/download/>`_  python environment with **Python 3.5 or 3.6**.

It is required to install some dependencies, hence run the following inside a ``conda`` environment:

.. code-block:: bash

    conda install -c simpleitk simpleitk
    conda install -c astra-toolbox astra-toolbox
    conda install -c conda-forge ipython numpy numexpr matplotlib astropy tifffile opencv scikit-image read-roi mkl_fft scipy six tqdm pywavelets

Then install NeuTomPy toolbox via ``pip``:

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
