# demogPosSelDadiScripts

This directory contains the scripts used in Dan Schrider, Alex Shanku, and Andy
Kern's analyses of the impact of positive selection on demographic inference
using dadi. In order to run these scripts after checking out this repository
you will have to install the following packages on your machine (which must be
running a Unix-like operating system):

1) python
2) scipy
3) numpy
4) dadi (version 1.6.3)
5) pyOpt
6) MPI
and 7) mpi4py

You then have to run the following command and place it in your ~/.profile file
before running the scripts (but without the < and > characters):

export PYTHONPATH=$PYTHONPATH:<insert full path to wherever you place this repository on your machine>

dadiFunctions.py contains several functions used by the other scripts which
perform the actual inference. Each of these  other scripts takes as its first
argument the path to a file with simulated population genetic data in ms-style
format. For our analyses we used discoal_multipop
(https://github.com/kern-lab/discoal_multipop). For singlePopEquilibrium.py
this is the only argument. For the other three, a second argument is required:
the number of cores that the optimization procedure will use. These three
scripts must be run using mpiexec. For example:

/home/schrider/anaconda/bin/mpiexec -n $ncores python singlePopGrowth.py $simulatedData $ncores

where $ncores is the number of cores you wish to use, and $simulatedData is the
path to the ms-formatted data set on which dadi will perform optimization.
