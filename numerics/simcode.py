from simulation import *
from joblib import Parallel
import pickle, gzip

num_cores = 24

results_0= [ Parallel(n_jobs=num_cores)(delayed(process_M)(M=2, J=int(4**J)) for k in range(1000)) for J in range(1,7) ]
print(0)
results_1 = [ Parallel(n_jobs=num_cores)(delayed(process_M)(M=4, J=int(4**J)) for k in range(1000)) for J in range(1,7) ]
print(1)
results_2 = [ Parallel(n_jobs=num_cores)(delayed(process_M)(M=8, J=int(4**J)) for k in range(1000)) for J in range(1,7) ]
print(2)
results_3 = [ Parallel(n_jobs=num_cores)(delayed(process_M)(M=16, J=int(4**J)) for k in range(1000)) for J in range(1,7) ]
print(3)
results_4 = [ Parallel(n_jobs=num_cores)(delayed(process_M)(M=16, J=int(4**J)) for k in range(1000)) for J in range(1,7) ]

with gzip.open('simulation_results_M.gz','wb') as f:
    pickle.dump([results_0,results_1,results_2,results_3,results_4],f)

results_0 = [ Parallel(n_jobs=num_cores)(delayed(process_q)(q=0, J=int(10**J)) for k in range(100)) for J in range(1,6) ]
print(0)
results_1 = [ Parallel(n_jobs=num_cores)(delayed(process_q)(q=1, J=int(10**J)) for k in range(100)) for J in range(1,6) ]
print(1)
import pickle, gzip
with gzip.open('simulation_results_q.gz','wb') as f:
    pickle.dump([results_0,results_1],f)
results_2 =  [ Parallel(n_jobs=num_cores)(delayed(process_q)(q=2, J=int(10**J)) for k in range(100)) for J in range(1,7) ]
print(2)
#results_3 = [ Parallel(n_jobs=num_cores)(delayed(process_q)(q=3, J=int(10**J)) for k in range(100)) for J in range(1,7) ]
#print(3)
#results_4 = [ Parallel(n_jobs=num_cores)(delayed(process_q)(q=4, J=int(10**J)) for k in range(100)) for J in range(1,7) ]
#print(4)
#results_5 = [ Parallel(n_jobs=num_cores)(delayed(process_q)(q=5, J=int(10**J)) for k in range(100)) for J in range(1,7) ]
#print(5)

with gzip.open('simulation_results_q.gz','wb') as f:
    pickle.dump([results_0,results_1,results_2],f)
