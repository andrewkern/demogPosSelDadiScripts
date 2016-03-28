import dadi
import numpy
import scipy
import pyOpt
import dadiFunctions
import Demographics1D
import sys

msFileName = sys.argv[1]

print "Calculating likelihood of constant population size model given data in %s.\nAll other arguments ignored." %(msFileName)

msFile = open(msFileName)

data = dadi.Spectrum.from_ms_file(msFile, average=False)
msFile.close()

ns=[200]
pts_l = [20, 30, 40]

func = Demographics1D.snm

func_ex = dadi.Numerics.make_extrap_func(func)

# Instantiate Optimization Problem 

model = func_ex("asdf", ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)

u = 1.2e-8
L = 50000000 # assuming we have 500 simulated loci that are 100 kb in length

popt=[]
print 'Log-likelihood:', ll_opt
print 'AIC:',(-2*ll_opt) + 2*len(popt)

#scaled estimates
theta0 = dadi.Inference.optimal_sfs_scaling(model, data)
print 'theta0=%s. Assuming u = %s:' %(theta0, u)
Nref= theta0 / u / L / 4
print 'Nref:',Nref
print ""
print "Ne"
print "%s" %(Nref)
sys.stderr.write("done with %s\n" %(msFileName))
