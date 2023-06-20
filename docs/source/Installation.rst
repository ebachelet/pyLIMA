.. _installation:


Installation
============


pip
---

The last stable version can be installed via pip:

.. code-block:: bash

    python -m pip install -U pip
    pip install -U pyLIMA
    

From source
-----------
    
You can also get the development version on GitHub:

.. code-block:: bash

    git clone https://github.com/ebachelet/pyLIMA.git
    pip install .

You can also setup a virtualenv.

Virtual Environments
--------------------

You can use a virtual environment to run pyLIMA (and to run other packages). 
`Here <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_ is a quick guide to virtual environements.

Here is the description using pip, procedure should be similar if your more familiar with anaconda.

Install virtualenv:

.. code-block:: bash

    pip install --U virtualenv

Create an environment, here we called it pyLIMA_env:

.. code-block:: bash
    
    python3 -m venv pyLIMA_env
 
Activate the environnement:

.. code-block:: bash

    source ~/envs/pyLIMA_env/bin/activate

Deactivate the environnement:

.. code-block:: bash
    
    deactivate


