Virtual environnement
=====================


We highly recommand to setup a virtual environnement to run pyLIMA. This is not mandatory however. 
`here <http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/>`_ is a quick guide to virtual environements.

Here is the description using pip, procedure should be similar if your more familiar with anaconda.

Install virtualenv:

**> pip install --user virtualenv**

close and reopen terminal, or virtualenv command may not work

[!!!!! WARNING !!!!! : SPHYNX is bugged for double hyphen. There is a double hyphen before user, not a single one!]

Create an environnement, here we called it pyLIMA:

**> virtualenv ~/envs/pyLIMA**
 
Activate the environnement:\

**> source ~/envs/pyLIMA/bin/activate**

If you ever want to deactiate the env, you can use:

**> deactivate**
