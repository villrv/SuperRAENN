=====
Usage
=====

SuperRAENN can be used on a dataset containing both spectroscopically labelled and unlabelled SNe. All events will be used to train the RAENN, while labelled events will be used to train the random forest. 

A minimal working example is shown below:

.. code-block:: bash

    superraenn-prep example_LCs/ example_meta_table.dat
    superraenn-raenn ./products/lcs.npz --n-epoch 10
    superraenn-extract products/lcs.npz
    superraenn-classify example_meta_table.dat --train --savemodel
    superraenn-classify example_meta_table.dat --modelfile ./products/model.sav

As shown, SuperRAENN needs two inputs: a directory of light curve files (in `SNANA <https://github.com/RickKessler/SNANA>` format) and a metatable containing the SN name, redshifts, SN types, best-guess at explosion time and Milky Way extinction.

-----------------------
Light Curve Data Format
-----------------------
Light curves used to run SuperRAENN use  `SNANA <https://github.com/RickKessler/SNANA>`_ text format. 

Below is an example light curve file::

    SURVEY: PS1MD
    SNID:  PSc000001
    IAUC:    UNKNOWN
    RA:        52.4530625  deg
    DECL:       -29.0749750  deg
    MWEBV: 0.0075 +- 0.0003 MW E(B-V)
    REDSHIFT_FINAL:  0.0713 +- 0.0010  (CMB)
    SEARCH_PEAKMJD: 55207.0
    FILTERS:    griz

    # ======================================
    # TERSE LIGHT CURVE OUTPUT
    #
    NOBS: 306
    NVAR:   7
    VARLIST: MJD  FLT FIELD   FLUXCAL   FLUXCALERR    MAG     MAGERR

    OBS: 55086.6 g NULL  -243.440 231.478 nan -1.032
    OBS: 55089.6 g NULL  -62.931 13.480 nan -0.233
    OBS: 55095.6 g NULL  -15.102 16.238 nan -1.167
    OBS: 55098.6 g NULL  -94.646 13.910 nan -0.160
    OBS: 55104.6 g NULL  -28.093 12.441 nan -0.481
    OBS: 55191.3 g NULL  -27.414 10.304 nan -0.408
    OBS: 55203.3 g NULL  1381.526 18.142 -12.851 0.014
    OBS: 55446.6 g NULL  -3.432 9.291 nan -2.939
    END:

---------------------
Metatable Data Format
---------------------

The metatable is necessary for SuperRAENN to understand which files are in the supervised vs. unsuperised portions of the data. The format must match that shown in the example file, shown below::

    # SN Redshift Type T_explosion MW(EBV)
    PSc000001 0.071 SNII 55207.0 0.008
    PSc000006 0.2308 SNIa 55207.0 0.008
    PSc010411 0.0740 SNIbc 55248.0 0.009
    PSc060270 0.9560 SLSN 55389.0 0.034
    PSc070763 0.2590 SNIIn 55424.0 0.037
    PSc000345 0.5446 - 55215.0 0.009
    PSc000349 0.4802 - 55215.0 0.027
    PSc000353 0.268 - 55215.0 0.016
    PSc000363 0.1964 - 55215.0 0.025
    PSc000418 0.6881 - 55216.0 0.020
