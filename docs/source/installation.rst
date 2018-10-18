============
Installation
============

NeuTomPy toolbox supports **Linux** and **Windows** 64-bit operating system.

First of all, install a `Conda <https://www.anaconda.com/download/>`_  python environment with **Python 3.5 or 3.6**.

It is required to install some dependencies, hence run the following inside a ``conda`` environment:

.. code-block:: bash

    conda install -c simpleitk simpleitk
    conda install -c astra-toolbox astra-toolbox
    conda install -c conda-forge numexpr matplotlib astropy tifffile opencv scikit-image read-roi tqdm pywavelets

Then install NeuTomPy toolbox via ``pip``:

.. code-block:: bash

    pip install neutompy


Update
------

To update a NeuTomPy installation to the latest version run:

.. code-block:: bash

    pip install neutompy --upgrade
