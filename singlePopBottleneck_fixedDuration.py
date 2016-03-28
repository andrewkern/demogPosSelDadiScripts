import dadi
import numpy
import scipy
import pyOpt
import dadiFunctions
import sys

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except:
    raise ImportError('mpi4py is required for parallelization')

msFileName, swarmSize = sys.argv[1:]
swarmSize = int(swarmSize)

msFile = open(msFileName)

data = dadi.Spectrum.from_ms_file(msFile, average=False)
msFile.close()

ns=[200]
pts_l = [20, 30, 40]

func = dadiFunctions.three_epoch_fixed_duration

paramsTxt =['nuB', 'nuF', 'TF']
upper_bound = [1.0, 10.0, 5000/(2.0*10000)]
lower_bound = [1/2000.0, 1/200.0, 100/(2.0*10000)]

#starting with random inital guesses of parameter values by perturbing the true values
p0 = [0.2, 2.0, 3000/(2*10000.0)]
p1 = dadi.Misc.perturb_params(p0, fold=0.5, lower_bound=lower_bound, upper_bound=upper_bound)

sys.stderr.write("working on %s\n%s\n%s\n" %(msFileName, paramsTxt, p1))

func_ex = dadi.Numerics.make_extrap_func(func)

# Instantiate Optimization Problem 

def objfunc(x):
    f = dadi.Inference._object_func(x, data, func_ex, pts_l, 
                                      lower_bound=lower_bound,
                                          upper_bound=upper_bound)
    g=[]
    fail = 0
    return f,g,fail

opt_prob = pyOpt.Optimization('dadi optimization',objfunc)
opt_prob.addVar('nuB','c',lower=lower_bound[0],upper=upper_bound[0],value=p1[0])
opt_prob.addVar('nuF','c',lower=lower_bound[1],upper=upper_bound[1],value=p1[1])
opt_prob.addVar('TF','c',lower=lower_bound[2],upper=upper_bound[2],value=p1[2])
opt_prob.addObj('f')

if myrank == 0:
    print opt_prob


#optimize
psqp = pyOpt.ALPSO(pll_type='DPM')
psqp.setOption('printOuterIters',1)
psqp.setOption('SwarmSize', swarmSize)
psqp.setOption('fileout', 0)
psqp(opt_prob)
print opt_prob.solution(0)

popt = numpy.zeros(len(p1))
for i in opt_prob._solutions[0]._variables:
    popt[i]= opt_prob._solutions[0]._variables[i].__dict__['value']

model = func_ex(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)

u = 1.2e-8
L = 50000000 # assuming we have 500 simulated loci that are 100 kb in length

if myrank == 0:
    print 'Optimized log-likelihood:', ll_opt
    print 'AIC:',(-2*ll_opt) + 2*len(popt)
    #scaled estimates
    theta0 = dadi.Inference.optimal_sfs_scaling(model, data)
    Nref= theta0 / u / L / 4

    print 'Nref:',Nref
    scaledParams = [Nref*popt[0],Nref*popt[1],2*Nref*popt[2],2*Nref]
    for i in range(len(paramsTxt)):
        print paramsTxt[i],':',str(scaledParams[i])
    print ""
    print repr(popt)

############### 
# Now refine the optimization using Local Optimizer
# Instantiate Optimizer (SLSQP) 
# Instantiate Optimizer (SLSQP)
slsqp = pyOpt.SLSQP()
slsqp.setOption('IPRINT', 0)
# Solve Problem (With Parallel Gradient)
if myrank == 0:
    print 'going for second optimization'

delta=1e-3
for i in range(len(popt)):
    if popt[i] <= lower_bound[i]:
        popt[i] = lower_bound[i]+delta
    elif popt[i] >= upper_bound[i]:
        popt[i] = lower_bound[i]-delta

slsqp(opt_prob.solution(0),sens_type='FD',sens_mode='pgc')
print opt_prob.solution(0).solution(0)
opt = numpy.zeros(len(p1))
for i in opt_prob._solutions[0]._solutions[0]._variables:
    popt[i]= opt_prob._solutions[0]._solutions[0]._variables[i].__dict__['value']

model = func_ex(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)
if myrank == 0:      
    print 'After Second Optimization'
    print 'Optimized log-likelihood:', ll_opt
    print 'AIC:',(-2*ll_opt) + 2*len(popt)

    #scaled estimates
    theta0 = dadi.Inference.optimal_sfs_scaling(model, data)
    print 'theta0=%s. Assuming u = %s:' %(theta0, u)
    Nref= theta0 / u / L / 4

    print 'Nref:',Nref
    scaledParams = [Nref*popt[0],Nref*popt[1],2*Nref*popt[2],2*Nref]
    for i in range(len(paramsTxt)):
        print paramsTxt[i],':',str(scaledParams[i])
    print ""
    print repr(popt)
    print "NeA\tneB\ttR\tNe0"
    print "%s\t%s\t%s\t%s" %(Nref, scaledParams[0], scaledParams[2], scaledParams[1])
    sys.stderr.write("done with %s\n" %(msFileName))
