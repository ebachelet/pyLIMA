Virtual Environments
======================


You can use a virtual environment to run pyLIMA (and to run other packages). 
`Here <http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/>`_ is a quick guide to virtual environements.

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
