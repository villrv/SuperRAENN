==========
superRAENN
==========

.. image:: https://zenodo.org/badge/253530456.svg
   :target: https://zenodo.org/badge/latestdoi/253530456

.. image:: https://img.shields.io/travis/villrv/superraenn.svg
        :target: https://travis-ci.org/villrv/superraenn

.. image:: https://img.shields.io/pypi/v/superraenn.svg
        :target: https://pypi.python.org/pypi/superraenn


SuperRAENN is an open-source Python 3.x package for the photometric classification of supernovae in the following categories: Type I superluminos supernovae, Type II, Type IIn, Type Ia and Type Ib/c. It is described in detail in Villar et al. (in prep.). SuperRAENN is optimized for use with complete (rather than realtime) light curves from the Pan-STARRS Medium Deep Survey. *Users will need to train the classifier on their own data for optimal results.*

* Free software: 3-clause BSD license
* Documentation is available (here)[https://superraenn.readthedocs.io/en/latest/]

Installation
--------

`superraenn` is available with `pip`:


``
pip install superraenn
``

For a development install, clone this directory:

``
git clone https://github.com/villrv/superraenn.git
cd superraenn
python setup.py develop
``
