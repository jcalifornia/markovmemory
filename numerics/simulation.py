#/usr/bin/env python3.6
import sys
if sys.version_info[0] != 3 or sys.version_info[1] < 6:
    print("This script requires Python version 3.6")
    sys.exit(1)

from numpy.random import dirichlet, choice
from itertools import product
from string import ascii_uppercase
from collections import defaultdict
from functools import partial
from scipy.special import gammaln, polygamma
import numpy as np
from numpy import log 
from numpy import inf
from joblib import Parallel, delayed
import multiprocessing
import argparse

np.seterr(divide='ignore')


def dict_to_mat(dictdata):
    return np.array([val for val in dictdata.values()])


def simulate(num_states, q, alpha, num_trajectories = 100, max_length = 100):
    # simulates trajectories up to a maximum size
    
    states = [m for m in ascii_uppercase[:num_states]]

    X0 = ["".join(x) for x in product(states,repeat=q) if states[-1] not in x[:-1] ]

    pad_states = set()
    
    for j in range(q-2):
        pad_states |= set(["".join(["0"]*(j+1) + ["A"] +[x[:-(j+2)]]) for x in X0])

    pad_states.add("".join(["0"]*(q-1)+["A"]))
    
    X = list(pad_states)
    X.extend(X0)
    p = dirichlet(alpha*np.ones(num_states),len(X)) # also pad with boundary conditions
    states = [m for m in ascii_uppercase[:num_states]]
    trajectories = []
    for j in range(num_trajectories):
        trajectories += ["".join(["0"]*(q-1)+["A"])]
        current_state = "A"
        while not current_state == states[-1] and len(trajectories[j]) < max_length+q-1:
            subchain = trajectories[j][-q:]
            r = choice(states,size=1,p=p[X.index(subchain)])
            trajectories[j]+= r[0]
            current_state = r[0]
        # de-pad the trajectories
        trajectories[j] = trajectories[j][(q-1):]
            
    return trajectories,states

def infer_model(trajectories,states,alpha,q):
    N = defaultdict(partial(np.zeros,len(states)))
    Ntot = defaultdict(int)
    J = len(trajectories)
    for j,trajectory in enumerate(trajectories):
        traj = "".join(['0']*(q) ) +trajectory
        for l in range(q,len(traj)):
            if q==0: x = ''
            else:
                x = traj[(l-q):(l)]
                if x[-1] == '0' and len(x)>1: continue
            m = states.index(traj[l])
            Ntot[x] += 1
            N[x][m] += 1
            
    probs = defaultdict(partial(np.zeros,len(states)))
    for key, val in N.items():
        probs[key] = (N[key]+alpha)/(Ntot[key] + alpha*len(states))
        
    return probs, N

def evaluate_models(trajectories,states, alpha=1, qbounds = (1,8)):
    WAIC1 = {}
    WAIC2 = {}
    DIC1 = {}
    DIC2 = {}
    AIC = {}

    kWAIC1 = {}
    kWAIC2 = {}
    kDIC1 = {}
    kDIC2 = {}

    LPPD = {}
    LPD = {}
    LOO = {}
    LPPDCV2 = {}

    N = {}
    N2 = {}

    for q in range(qbounds[0],qbounds[1]+1):
        #print(f"Fitting models of {q} states of hysteresis")
        N[q] = defaultdict(partial(np.zeros,len(states)))
        N2[q] = defaultdict(partial(np.zeros,len(states))) # sums for the first half of trajectories

        cumulative_lppd = 0
        cumulative_kWAIC2 = 0
        cumulative_loo = 0
        cumulative_cv2 = 0

        J = len(trajectories)

        for j,trajectory in enumerate(trajectories):
            trajectory = "".join(['0']*(q) ) +trajectory
            for l in range(q,len(trajectory)):
                if q==0: x = ''
                else:
                    x = trajectory[(l-q):(l)]
                    if x[-1] == '0' and len(x)>1: continue
                m = states.index(trajectory[l])
                N[q][x][m] += 1
                if j<J/2:
                    N2[q][x][m] += 1
            
        for j,trajectory in enumerate(trajectories):
            trajectory = "".join(['0']*(q) ) +trajectory
            Nj = defaultdict(partial(np.zeros,len(states)))
            for l in range(q,len(trajectory)):
                if q==0: x = ''
                else:
                    x = trajectory[(l-q):(l)]
                    if x[-1] == '0': continue
                Nj[x][states.index(trajectory[l])] += 1

            thiscountmatrix = []
            globalcountmatrix = []
            halfglobalcountmatrix = []
            for key, val in Nj.items():
                thiscountmatrix += [val] # counts for this trajectory
                globalcountmatrix += [ N[q][key]] # total counts across all trajectories
                halfglobalcountmatrix += [ N2[q][key]]

            # recast to numpy
            thiscountmatrix = np.array(thiscountmatrix,dtype=np.int64)  # N^{(j)}_\bx
            globalcountmatrix = np.array(globalcountmatrix,dtype=np.int64) # N_\bx

            if j>=J/2: # want the other half
                # want the first half if our trajectory is from the second half
                halfglobalcountmatrix = np.array(halfglobalcountmatrix, dtype=np.int64)
            else:
                # want the second half if our trajectory is from the first half
                halfglobalcountmatrix = globalcountmatrix - np.array(halfglobalcountmatrix, dtype=np.int64)

            globalcountmatrix_loo = globalcountmatrix - thiscountmatrix
            
            
            this_log_gammas = gammaln(thiscountmatrix + globalcountmatrix + alpha)
            global_log_gammas = gammaln(globalcountmatrix+alpha)
            global_log_gammas_loo = gammaln(globalcountmatrix_loo + alpha)

            # do computations on the individual trajectory level
            try:
                this_row_sums = np.sum(thiscountmatrix+globalcountmatrix+alpha, axis = 1) # total number of times a history is seen locally
            except BaseException as e:
                pass
            
            global_row_sums = np.sum(globalcountmatrix+alpha, axis = 1) # total number of times a history is seen globally
            global_row_sums_loo = np.sum(globalcountmatrix_loo + alpha, axis=1)

            half_global_row_sums = np.sum(halfglobalcountmatrix, axis =1)
            half_global_log_gammas = gammaln(halfglobalcountmatrix + alpha)

            half_global_log_gammas_cv2 = gammaln(halfglobalcountmatrix + thiscountmatrix + alpha)
            half_global_row_sums_cv2 = np.sum(halfglobalcountmatrix + thiscountmatrix + alpha, axis =1)


            """
            The lppd can be computed exactly
            \begin{align}
            \sum_{\bx}\log\mathbb{E}_{\bp_\bx\vert\bN_\bx}  \left[     \Pr\left(\mathbf{N}^{(j)}_{\bx} \mid \bp_\bx \right) \right]  \nonumber\\
            &= \sum_j\sum_{\mathbf{x}} \log \mathbb{E}_{\bp_\bx\vert\bN_\bx} \left(% \frac{N_{\mathbf{x}}!}{\mathbf{N}_{\mathbf{x}}!}
             \prod_{m=1}^{M} p_{\bx,m}^{N^{(j)}_{\bx,m}} \right) \nonumber\\
            &=\sum_j \sum_{\bx}  \log\left(  \frac{B(\bN_\bx +\bN_{\bx}^{(j)} +\balpha)}{B(\bN_\bx +\balpha)} \right).
            \end{align}
            """

            # this trajectory's contribution to the lppd
            partial_lppd = np.sum(np.sum(this_log_gammas, axis = 1) - gammaln(this_row_sums) - \
                    np.sum(global_log_gammas, axis=1) + gammaln(global_row_sums) )
            partial_loo = np.sum(np.sum(global_log_gammas, axis = 1) - gammaln(global_row_sums) - \
                    np.sum(global_log_gammas_loo, axis=1) + gammaln(global_row_sums_loo) )

            partial_cv2 = np.sum(np.sum(half_global_log_gammas_cv2, axis = 1) - gammaln(half_global_row_sums_cv2) - \
                        np.sum(half_global_log_gammas, axis=1) + gammaln(half_global_row_sums+this_log_gammas.shape[1]*alpha) )

            partial_kWAIC2 = np.sum(thiscountmatrix**2*polygamma(1,alpha+globalcountmatrix)) - \
                np.sum(np.sum(thiscountmatrix,axis=1)**2*polygamma(1, np.sum(globalcountmatrix + alpha,axis=1)))


            cumulative_lppd += partial_lppd
            cumulative_loo += partial_loo
            cumulative_kWAIC2 += partial_kWAIC2
            cumulative_cv2 += partial_cv2

        # AIC and DIC don't need individual trajectories
        totalcountmatrix = dict_to_mat(N[q])

        totalrowsums = np.sum(totalcountmatrix,axis=1)
        logp_MLE = np.log(totalcountmatrix/totalrowsums[:, None])
        logp_MLE[np.isneginf(logp_MLE)] = 0

        Nlogp_MLE =  np.sum(totalcountmatrix*logp_MLE)
        Nlogp2 = np.sum(totalcountmatrix*log( (totalcountmatrix + alpha) / np.sum(totalcountmatrix + alpha,axis = 1)[:,None]))

        expected_log_posterior = np.sum(totalcountmatrix*(polygamma(0,alpha+totalcountmatrix) - polygamma(0,np.sum(totalcountmatrix + alpha,axis=1))[:,None]))
        kDIC1[q] = 2*np.sum( totalcountmatrix*(np.log(totalcountmatrix+alpha) - np.log(np.sum(totalcountmatrix + alpha,axis=1))[:,None])) - 2*expected_log_posterior

        kDIC2[q] = 2*np.sum( totalcountmatrix**2*polygamma(1,alpha+totalcountmatrix)) \
                   -2*np.sum( totalrowsums**2*polygamma(1,totalrowsums + totalcountmatrix.shape[1]*alpha) )

        kWAIC1[q] = 2*cumulative_lppd -2*expected_log_posterior
        kWAIC2[q] = cumulative_kWAIC2
        DIC1[q] = -2*Nlogp2 +2*kDIC1[q]
        DIC2[q] = -2*Nlogp2 +2*kDIC2[q]
        WAIC1[q] = -2*(cumulative_lppd - kWAIC1[q])
        WAIC2[q] = -2*(cumulative_lppd - kWAIC2[q])
        kAIC = totalcountmatrix.shape[0]*totalcountmatrix.shape[1]
        AIC[q] = -2*Nlogp_MLE + 2*kAIC
        LPPD[q] = -2*(cumulative_lppd)
        LOO[q] = -2*(cumulative_loo)

        log2gammas = gammaln(2*totalcountmatrix + alpha)
        log1gammas = gammaln(totalcountmatrix + alpha)

        LPD[q] = -2*(np.sum(np.sum(log2gammas, axis = 1) - gammaln(2*totalrowsums + alpha) - \
                    np.sum(log1gammas, axis=1) + gammaln(totalrowsums + alpha) ))

        LPPDCV2[q] = -2*(cumulative_cv2)


    return {"WAIC1": WAIC1, "WAIC2": WAIC2, "LOO": LOO, "LPPDCV2": LPPDCV2,
            "DIC1": DIC1, "DIC2": DIC2, "AIC": AIC,"LPPD": LPPD, "LPD": LPD,
            "kWAIC1": kWAIC1, "kWAIC2" : kWAIC2, "kDIC1": kDIC1, "kDIC2": kDIC2 }

def process_q(q=2,J=100, M = 8, L=100,qbounds = (1,6)):
    X, X0 = simulate(num_states=M, q=q, alpha=1, num_trajectories=J, max_length=L)
    info = evaluate_models(X, X0, qbounds= qbounds)
    return(info)

def main():
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser()
    parser.add_argument("-q", "--hysteresis", dest="q",
                        help="Real hysteresis", metavar="HYST")

    results = parser.parse_args()
    hysteresis = results.q if results.q is not None else 2

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(process_q)() for q in range(10000))

    histograms = defaultdict(partial(np.zeros,5))
    criteria = defaultdict(list)

    for info in results:
        for key, val in info.items():
            best = np.argmin([v for v in val.values()])
            qvals = [v for v in val.keys()]
            histograms[key][best] += 1


    print(histograms)

    # output the results as a wide table

    #print("Total observed transitions:" + str(np.sum([len(x) for x in X])))
    #for key, val in info.items():
    #    best = np.argmin([v for v in val.values()])
    #    qvals = [v for v in val.keys()]
    #    print(key + "(least is q=" + str(qvals[best]) +")")

    #    for q, vals in val.items():
    #        print(f'{q}: {vals:0.3f}')


if __name__ == '__main__':
    main()