import numpy as np
import numba as nb
from .fxns.numba_math import invpsi, psi, gammaln, rev_eye
from .fxns.statistics import dkl_dirichlet


def initialize(data,N,depth_vec,prod_states,guess):
    dat = (data.T).flatten()
    depth = len(depth_vec) + 1
    count = np.ones(depth)
    active = np.zeros((depth,1))
    active[0] = 1
    path = np.zeros((depth,np.prod(depth_vec)))

    for i in range(np.prod(depth_vec)):
        path[:,i] = count
        for j in range(len(count)-1):
            if count[j] % depth_vec[j] == 0 and count[j]!=1 and active[j]:
                active[j+1] = 1
            else:
                active[j+1] = 0
        count += active.T[0]

    path = np.fliplr(path.T).T
    pathnum = path.shape[1]
    depth_vector = [1] + depth_vec + [prod_states]

    test = [[0 for j in range(int(path[i-1,-1]))] for i in range(depth,0,-1)]
    dim = np.flip([len(i) for i in test])
    #print(test)
    #print(dim)
    pi = [[0] * i for i in dim]
    tm = [[0] * i for i in dim]
    exit = [[0] * i for i in dim]
    tm_exit_alpha = [[0] * i for i in dim]
    pi_alpha = [[0] * i for i in dim]
    carryover = []

    for i in range(depth-1,-1,-1):
        for j in range(int(path[i,-1])):
            pi[i][j] = np.ones((1,depth_vector[i+1])) / depth_vector[i+1]

            # can either exit up or within a family; hence such normalization
            #tm[i][j] = np.eye(depth_vector[i+1]) + (np.ones((depth_vector[i+1], depth_vector[i+1])))
            #tm[i][j] = np.ones((depth_vector[i+1], depth_vector[i+1]))

            if i != depth-1:
                tm[i][j] = np.eye(depth_vector[i+1])*(0.1 + 1*np.random.rand(depth_vector[i+1],
                          depth_vector[i+1]))/ 5 + rev_eye(depth_vector[i+1])
                #tm[i][j] = (np.eye(depth_vector[i+1])
                            #+ np.random.rand(depth_vector[i+1])*rev_eye(depth_vector[i+1])/10)
            else:
                tm[i][j] = rev_eye(depth_vector[i+1])*(0.1 + 1*np.random.rand(depth_vector[i+1],
                           depth_vector[i+1])) / 5 + np.eye(depth_vector[i+1])
                #tm[i][j] = (np.ones((depth_vector[i+1],depth_vector[i+1]))
                            #+ np.random.rand(depth_vector[i+1])*np.eye(depth_vector[i+1])/5)

            #print(tm[i][j])
            exit[i][j] = tm[i][j].min(0) / 10.
            temp = np.concatenate((tm[i][j], np.array([exit[i][j]])),axis=0)
            tm[i][j] /= np.sum(temp,0)
            exit[i][j] /= np.sum(temp,0)

            # using Minka's EM routine for the Dirichlet prior
            if i == depth-1:
                bound = len(dat) / path[i,-1] / 20
            else:
                bound = carryover / 20

            temp = np.concatenate((tm[i][j], np.array([exit[i][j]])),axis=0)
            temp2 = np.diag(-np.log(tm[i][j]))
            temp2 = 1 - temp2 / np.sum(temp2)
            guess_tm_exit = (bound / path[i,-1] * temp) * temp2.T
            guess_pi = (bound / path[i,-1] * pi[i][j]).reshape(-1,1)
            tm_exit_alpha[i][j] = getprior(bound,guess_tm_exit,temp)
            pi_alpha[i][j] = getprior(bound,guess_pi,pi[i][j].reshape(-1,1))

        if i != 0:
            for k in range(int(path[i-1,-1])):
                carryover = 0
                child = np.unique(path[i,path[i-1,:]==k+1])
                for n in range(len(child)):
                    carryover += np.sum(tm_exit_alpha[i][j][-1,:])

        # FIX CARRYOVER AND MAKE IT A LIST

    mixture_mix = np.ones((depth_vector[-1],1)) / depth_vector[-1]
    mixture_alpha = mixture_mix * len(dat) / 10
    a = np.ones((depth_vector[-1],1))
    b = np.var(dat, ddof=1) / depth_vector[-1]**2. * np.ones((depth_vector[-1],1))
    beta = 1e2 * np.ones((depth_vector[-1],1))
    mu = guess.reshape(-1,1).copy()

    '''
        if i != 1:
            for k in range(1,int(path[i-2,-1])+1):
                carryover = 0
                child = np.unique(path[i-1,path[i-2,:]==k])
                for n in range(len(child)):
                    carryover += np.sum(tm_exit_alpha_dummy[-1][-1,:])
    '''
    return pathnum, depth, path, depth_vector, pi, tm, exit, mixture_mix, mixture_alpha, a, b, beta, mu, \
           tm_exit_alpha, pi_alpha

def getprior(bound,guess,tm):
# based on Tom Minka's technical paper

    tolerance = 1
    converged = 0
    out = guess
    newout = out.copy()

    while converged != 1:
        for i in range(np.size(tm,1)):
            newout[:,i] = invpsi(psi(np.sum(out[:,i])) + np.log(tm[:,i]))
        newout *= bound / np.sum(np.sum(newout))
        if np.sum(np.sum(np.abs(newout-out))) < tolerance:
            converged = 1
        out = newout.copy()

    return out

@nb.njit
def gauss_prob_evaluate(data, post_a, post_b, post_beta, post_mu):
    a_t = post_a.flatten()
    b_t = post_b.flatten()
    beta_t = post_beta.flatten()
    m_t = post_mu.flatten()

    conprob = -np.log(2. * np.pi) + psi(a_t) - np.log(b_t) - (a_t / (2. * b_t) \
        * (1. / beta_t + m_t**2)) + a_t / (2 * b_t) * (-data**2 + 2 * post_mu * data).T
    conprob = np.exp(conprob.T)
    for i in range(conprob.shape[0]):
        x = np.where(conprob[i] == 0)
        conprob[x] = 2.22044604925e-16
    conprob *= 1. / np.sum(conprob,0)

    return conprob.T


def dirichlet_update(update_alpha,hyper_alpha,phi,bound):
    #print(hyper_alpha.shape,phi.shape)
    #if phi.ndim == 1:
    #pre-programmed the message into the forward-backward algorithm
    if hyper_alpha.shape[1] == 1:
        message = np.sum(phi,0).reshape(-1,1)
        phi = (hyper_alpha - 1.) + message
        update_alpha = phi + 1.
    else:
        update_alpha = phi + 1. + hyper_alpha

    # taken from Winn's thesis
    update_mix = np.empty_like(update_alpha)
    for i in range(update_alpha.shape[1]):
        update_mix[:,i] = np.exp(psi(update_alpha[:,i]) - psi(np.sum(update_alpha[:,i])))
    update_mix *= 1. / np.sum(update_mix,0)

    # the transition matrix is put on its side to acquiesce to the dimensions of the mixture components
    lowerbound = bound

    for i in range(phi.shape[1]):
        lowerbound = lowerbound - dkl_dirichlet(update_alpha[:,i], hyper_alpha[:,i])

    return update_alpha, update_mix, lowerbound


def gauss_update(data,update_a,update_b,update_beta,update_mu,hyper_a,hyper_b,hyper_beta,hyper_mu,
                 hyper_mixture_mix,conprob,index,bound):
    i = index

    # mu update
    message_mu = np.zeros((2,len(data)))
    message_mu[0,:] = data.flatten().T * hyper_a[i,:] / hyper_b[i,:]
    message_mu[1,:] = -np.ones((1,len(data))) * hyper_a[i,:] / (2. * hyper_b[i,:])
    message_mu *= conprob
    phi_mu = np.array([hyper_beta[i,:].dot(hyper_mu[i,:]), (-hyper_beta[i,:] / 2.)[0]]) \
                + np.sum(message_mu,1)
    phi_mu = phi_mu.reshape(-1,1)
    update_beta[i,:] = -2. * phi_mu[1,:]
    update_mu[i,:] = phi_mu[0,:] / update_beta[i,:]

    # precision update
    message_prec = np.zeros((2,len(data)))
    message_prec[0,:] = -0.5 * (data.flatten().T**2 - 2. * data.flatten().T * update_mu[i,:] \
                                 + update_mu[i,:]**2 + 1 / update_beta[i,:])
    message_prec[1,:] = np.ones((1,len(data))) / 2.
    message_prec *= conprob
    phi_prec = np.array([-hyper_b[i,:][0], hyper_a[i,:][0] - 1.]) + np.sum(message_prec,1)
    phi_prec = phi_prec.reshape(-1,1)
    update_b[i,:] = -phi_prec[0,:]
    update_a[i,:] = phi_prec[1,:] + 1.

    # lowerbound calculation
    lowerbound = -(np.dot(np.array([(update_beta[i,:] * update_mu[i,:] - hyper_beta[i,:] * hyper_mu[i,:])[0], \
                                   (-update_beta[i,:] / 2. + hyper_beta[i,:] / 2.)[0]]), \
                          np.array([update_mu[i,:],update_mu[i,:]**2 + 1 / update_beta[i,:]]).reshape(-1,1)) + \
                 0.5 * (np.log(update_beta[i,:]) - update_beta[i,:] * update_mu[i,:]**2 - \
                 np.log(hyper_beta[i,:]) + hyper_beta[i,:] * hyper_mu[i,:]**2))
    lowerbound = bound - hyper_mixture_mix[i,:] * (lowerbound + np.dot(np.array([(-update_b[i,:] + \
                                                   hyper_b[i,:])[0], (update_a[i,:] - hyper_a[i,:])[0]]), \
                                                   np.array([(update_a[i,:] / update_b[i,:])[0], \
                                                  (psi(update_a[i,:]) - \
                                                   np.log(update_b[i,:]))[0]]).reshape(-1,1)) + update_a[i,:] \
                       * np.log(update_b[i,:]) - gammaln(update_a[i,:]) \
                       - hyper_a[i,:] * np.log(hyper_b[i,:]) + gammaln(hyper_a[i,:]))

    return update_mu, update_beta, update_a, update_b, lowerbound[0]

def fb_activation(depth, path, depth_num, pi, tm, exit, conprob, T):
    scale = np.zeros((T+1, 1))

    test = [[0 for j in range(int(path[i-1,-1]))] for i in range(depth,0,-1)]
    dim = np.flip([len(i) for i in test])

    alpha_entry = [[0] * i for i in dim]
    alpha_exit = [[0] * i for i in dim]
    beta_entry = [[0] * i for i in dim]
    beta_exit = [[0] * i for i in dim]

    # alpha boundary conditions
    for i in range(depth):
        for j in range(int(path[i,-1])):
            alpha_entry[i][j] = np.ones((T,len(pi[i][j][0])))
            alpha_exit[i][j] = np.ones((T,len(pi[i][j][0])))
            beta_entry[i][j] = np.ones((T,len(pi[i][j][0])))
            beta_exit[i][j] = np.ones((T,len(pi[i][j][0])))

            if i == 0:
                alpha_entry[i][j][0] = pi[i][j][0]

            else:
                parent = int(np.unique(path[i-1,path[i,:] == j+1])[0])
                ca = ((j+1) % depth_num[i]) + depth_num[i] * (((j+1) % depth_num[i]) == 0)
                alpha_entry[i][j][0] = alpha_entry[i-1][parent-1][0][ca-1] * pi[i][j][0]

    # forward pass
    for time in range(1,T):
        for i in range(depth, 0, -1):
            for j in range(int(path[i-1,-1])):
                if i == depth:
                    alpha_exit[i-1][j][time-1] = alpha_entry[i-1][j][time-1] * conprob[time-1]
                    scale[time-1][0] += np.sum(alpha_exit[i-1][j][time-1])
                else:
                    if i != 1:
                        child = np.unique(path[i,path[i-1,:]==j+1])
                        for k in range(len(child)):
                            alpha_exit[i-1][j][time-1][k] = np.sum(alpha_exit[i][int(child[k])-1][time-1] * \
                                                                   exit[i][int(child[k])-1])
                    else:
                        for k in range(int(path[i,-1])):
                            alpha_exit[i-1][j][time-1][k] = np.sum(alpha_exit[i][k][time-1] * exit[i][k])
            if i == depth:
                for j in range(int(path[i-1,-1])):
                    alpha_exit[i-1][j][time-1] /= scale[time-1]


        for i in range(depth):
            if i == 0:
                for j in range(int(path[i,-1])):
                    alpha_entry[i][j][time] = np.dot(alpha_exit[i][j][time-1], tm[i][j].T)
            else:
                for j in range(int(path[i,-1])):
                    parent = np.unique(path[i-1,path[i,:]==j+1])[0]
                    ca = ((j+1) % depth_num[i]) + depth_num[i] * (((j+1) % depth_num[i]) == 0)
                    alpha_entry[i][j][time] = alpha_entry[i-1][int(parent-1)][time][ca-1] * pi[i][j] \
                                                    + np.dot(alpha_exit[i][j][time-1], tm[i][j].T)

    # forward pass clean-up
    time = T
    for i in range(depth, 0, -1):
        for j in range(int(path[i-1,-1])):
            if i == depth:
                alpha_exit[i-1][j][time-1] = alpha_entry[i-1][j][time-1] * conprob[time-1]
                scale[time-1][0] += np.sum(alpha_exit[i-1][j][time-1])
            else:
                child = np.unique(path[i,path[i-1,:]==i])
                for k in range(len(child)):
                    alpha_exit[i-1][j][time-1][k] = np.sum(alpha_exit[i][int(child[k])-1][time-1] * \
                                                           exit[i][int(child[k])-1])
        if i == depth:
            for j in range(int(path[i-1,-1])):
                alpha_exit[i-1][j][time-1] /= scale[time-1]

    #probability of the FDT state
    scale[time] = np.sum(alpha_exit[0][0][T-1] * exit[0][0])

    # beta variable boundary conditions
    beta_exit[0][0][T-1] = exit[0][0] / scale[T]
    for i in range(1, depth):
        for j in range(int(path[i,-1])):
            parent = np.unique(path[i-1,path[i,:]==j+1])[0]
            ca = ((j+1) % depth_num[i]) + depth_num[i] * (((j+1) % depth_num[i]) == 0)
            beta_exit[i][j][time-1] = beta_exit[i-1][int(parent)-1][T-1][ca-1] * exit[i][j]

    # backward pass
    for time in range(T-1, 0, -1):
        for i in range(depth, 0, -1):
            for j in range(int(path[i-1,-1])):
                if i == depth:
                    beta_entry[i-1][j][time] = beta_exit[i-1][j][time] * conprob[time] / scale[time]
                else:
                    child = np.unique(path[i,path[i-1,:]==i])
                    for k in range(len(child)):
                        beta_entry[i-1][j][time][k] = np.sum(beta_entry[i][int(child[k])-1][time] * \
                                                             pi[i][int(child[k])-1])

        for i in range(depth):
            for j in range(int(path[i,-1])):
                if i == 0:
                    beta_exit[i][j][time-1] = np.dot(beta_entry[i][j][time], tm[i][j])
                else:
                    parent = np.unique(path[i-1,path[i,:]==j+1])[0]
                    ca = ((j+1) % depth_num[i]) + depth_num[i] * (((j+1) % depth_num[i]) == 0)
                    beta_exit[i][j][time-1] = beta_exit[i-1][int(parent)-1][time-1][ca-1] * exit[i][j] + \
                                                np.dot(beta_entry[i][j][time], tm[i][j])

    # backward pass cleanup
    time = 0
    for i in range(depth, 0, -1):
        for j in range(int(path[i-1,-1])):
            if i == depth:
                beta_entry[i-1][j][time] = beta_exit[i-1][j][time] * conprob[time] / scale[time]
            else:
                child = np.unique(path[i,path[i-1,:]==i])
                for k in range(len(child)):
                    beta_entry[i-1][j][time][k] = np.sum(beta_entry[i][int(child[k])-1][time] * \
                                                         pi[i][int(child[k])-1])


    # precursor sufficient statistics
    # from here, the forward and backward variables are invisible & will be plugged directly int0
    # the dirichlets, etc.
    test = [[0 for j in range(int(path[i-1,-1]))] for i in range(depth,0,-1)]
    dim = np.flip([len(i) for i in test])

    out_a = [[0] * i for i in dim]
    out_pi = [[0] * i for i in dim]
    out_exit = [[0] * i for i in dim]
    out_gamma = [0] * dim[-1]

    for i in range(depth-1,-1,-1):
        for j in range(int(path[i,-1])):
            out_a[i][j] = np.ones((depth_num[i+1],depth_num[i+1]))
            out_pi[i][j] = np.ones((1,depth_num[i+1]))
            out_exit[i][j] = np.ones((1,depth_num[i+1]))

    for i in range(depth):
        for j in range(int(path[i,-1])):
            for k in range(depth_num[i+1]):
                for m in range(depth_num[i+1]):
                    out_a[i][j][m][k] = np.sum(alpha_exit[i][j][0:-1][:,k].reshape(-1,1) * tm[i][j][m][k] * \
                                               beta_entry[i][j][1:][:,m].reshape(-1,1))
            if i != 0:
                parent = np.unique(path[i-1,path[i,:]==j+1])[0]
                ca = ((j+1) % depth_num[i]) + depth_num[i] * (((j+1) % depth_num[i]) == 0)
                out_pi[i][j] = alpha_entry[i][j][0] * beta_entry[i][j][0] + \
                                                      np.sum(alpha_entry[i-1][int(parent)-1][1:].T[ca-1] * \
                                                      (beta_entry[i][j][1:] * pi[i][j]).T, 1)
                out_exit[i][j] = alpha_exit[i][j][T-1] * beta_exit[i][j][T-1] + \
                                                         np.sum(beta_exit[i-1][int(parent)-1][1:].T[ca-1] * \
                                                         (alpha_exit[i][j][1:] * exit[i][j]).T, 1)
            else:
                out_pi[i][j] = alpha_entry[i][j][0] * beta_entry[i][j][0]
                out_exit[i][j] = alpha_exit[i][j][T-1] * beta_exit[i][j][T-1]

    for i in range(int(path[-1,-1])):
        out_gamma[i] = alpha_exit[-1][i] * beta_exit[-1][i]

    # likelihood function
    like = np.sum(np.log(scale))

    return out_a, out_pi, out_exit, out_gamma, like

def viterbi_activation(depth, path, depth_num, pi, tm, mu, exit, conprob, T):
    # initialize variables
    test = [[0 for j in range(int(path[i-1,-1]))] for i in range(depth,0,-1)]
    dim = np.flip([len(i) for i in test])

    dE = [[0] * i for i in dim]
    dB = [[0] * i for i in dim]
    psiB = [[0] * i for i in dim]
    psiE = [[0] * i for i in dim[:-1]]

    # alpha boundary conditions
    for i in range(depth):
        for j in range(int(path[i,-1])):
            dE[i][j] = np.ones(len(pi[i][j][0]))
            dB[i][j] = np.ones(len(pi[i][j][0]))
            psiB[i][j] = np.ones((T,len(pi[i][j][0])))
            if i != depth-1:
                psiE[i][j] = np.ones((T,len(pi[i][j][0])))

    z = np.zeros((T,depth+1))
    d = np.ones((T,1))

    # entry probability
    dB[0] = np.log(pi[0])[0]
    for i in range(1,depth):
        for j in range(int(path[i,-1])):
            parent = np.unique(path[i-1,path[i,:]==j+1])[0]
            ca = ((j+1) % depth_num[i]) + depth_num[i] * (((j+1) % depth_num[i]) == 0)
            dB[i][j] = dB[i-1][int(parent)-1][ca-1] + np.log(pi[i][j][0])

    for time in range(1,T+1):
        # pull in the state designations from FBA algorithm
        for j in range(int(path[-1,-1])):
            dE[depth-1][j] = dB[depth-1][j] + np.log(conprob[j][time-1])

        # make a map of where the system would go were there a vertical transition
        for i in range(depth-1, 0, -1):
            for j in range(int(path[i-1,-1])):
                child = np.unique(path[i,path[i-1,:]==j+1])
                for k in range(len(child)):
                    prob = dE[i][int(child[k])-1] + np.log(exit[i][int(child[k])-1])
                    if all(x < 1e-5 for x in np.diff(prob)):
                        b = 0
                    else:
                        b = np.argmax(prob)
                    a = prob[b]
                    psiE[i-1][j][time-1][k] = b + 1
                    dE[i-1][j][k] = a

        # make a map for everything else
        if time != T:
            for k in range(int(path[1,-1])):
                prob = dE[0][0] + np.log(tm[0][0][:,k])
                if all(x < 1e-5 for x in np.diff(prob)):
                    b = 0
                else:
                    b = np.argmax(prob)
                a = prob[b]
                psiB[0][0][time][k] = b + 1
                dB[0][0][k] = a
            for i in range(1,depth):
                for j in range(int(path[i,-1])):
                    for k in range(len(pi[i][j][0])):
                        parent = np.unique(path[i-1,path[i,:]==j+1])[0]
                        ca = ((j+1) % depth_num[i]) + depth_num[i] * (((j+1) % depth_num[i]) == 0)
                        prob = (dE[i][j] + np.log(tm[i][j][:,k])).tolist()
                        prob.append(dB[i-1][int(parent)-1][ca-1] + np.log(pi[i][j][0][k]))
                        if all(x < 1e-5 for x in np.diff(prob)):
                            b = 0
                        else:
                            b = np.argmax(prob)
                        a = prob[b]
                        psiB[i][j][time][k] = b + 1
                        dB[i][j][k] = a

    # exiting the F1T state
    prob = dE[0][0] + np.log(exit[0][0])

    # backtracking
    b = np.argmax(prob)
    phi = [1, b+1]
    for i in range(1,depth):
        ca = (phi[i] % depth_num[i]) + depth_num[i] * ((phi[i] % depth_num[i]) == 0)
        phi.append(int(depth_num[i] * (phi[i] - 1) + psiE[i-1][phi[i-1]-1][T-2][int(ca)-1]))


    ca = (phi[depth] % depth_num[depth]) + depth_num[depth] * ((phi[depth] % depth_num[depth]) == 0)
    z[T-1,:] = phi
    meanpath = np.zeros((T,1))
    meanpath[T-1] = mu[int(ca)-1][0]
    d[T-1,:] = 1

    for time in range(T-1, 0, -1):
        for i in range(depth, 1, -1):
            ca = (phi[i] % depth_num[i]) + depth_num[i] * ((phi[i] % depth_num[i]) == 0)
            #print(phi[i-1]-1,type(phi[i-1]))
            if int(psiB[i-1][phi[i-1]-1][time][int(ca)-1]) <= depth_num[i]:
                d[time-1] = i+1
                phi[i] = int(depth_num[i] * (phi[i-1] - 1) + psiB[i-1][phi[i-1]-1][time][int(ca)-1])
                break
        for i in range(int(d[time-1]),depth):
            ca = (phi[i] % depth_num[i]) + depth_num[i] * ((phi[i] % depth_num[i]) == 0)
            phi[i] = int(depth_num[i] * (phi[i] - 1) + psiB[i-1][phi[i-1]-1][time][int(ca)-1])
            #print(phi[i],type(phi[i]))
        ca = (phi[depth] % depth_num[depth]) + depth_num[depth] * ((phi[depth] % depth_num[depth]) == 0)
        meanpath[time-1] = mu[int(ca)-1][0]
        z[time-1,:] = phi

    return z, meanpath

def vmp_hhmm(data,depth_vec,prod,max_iter,tol,restarts,initguess):
    #n = numstates
    N = len(data)
    depth = len(depth_vec) + 1
    fulldata = data.flatten()
    ev = np.NINF

    for it in range(restarts+1):
        print("Restart {}".format(it+1))

        stuff = initialize(data,N,depth_vec,prod,initguess)

        post_pathnum, post_depth, post_path, post_depth_num, post_pi, post_tm, post_exit, post_mixture_mix, \
        post_mixture_alpha, post_a, post_b, post_beta, post_mu, post_tm_exit_alpha, post_pi_alpha = \
        stuff

        prior_pathnum, prior_depth, prior_path, prior_depth_num, prior_pi, prior_tm, prior_exit, \
        prior_mixture_mix, prior_mixture_alpha, prior_a, prior_b, prior_beta, prior_mu, prior_tm_exit_alpha, \
        prior_pi_alpha = stuff

        converged = False
        count = 1
        gamma_all = np.zeros((len(fulldata),prod))
        record_exit = []
        record_pi = []
        record_tm = []
        record_evidence = []

        while converged == False:
            print('Starting Iteration %d'%(count))
            tempev = np.zeros((len(data),1))
            Ts = [None] * len(data)
            test = [[0 for j in range(int(post_path[i-1,-1]))] for i in range(post_depth,0,-1)]
            dim = np.flip([len(i) for i in test])

            acc_a = [[0] * i for i in dim]
            acc_exit = [[0] * i for i in dim]
            acc_pi = [[0] * i for i in dim]

            # run the forward-backward-activation algorithm on all time series
            for index in range(N):
                temp = data[index]
                conprob = gauss_prob_evaluate(temp, post_a, post_b, post_beta, post_mu)
                out_a, out_pi, out_exit, out_gamma, like = \
                fb_activation(post_depth, post_path, post_depth_num, post_pi, post_tm, post_exit,
                              conprob, len(temp))
                Ts[index] = [out_a, out_pi, out_exit, out_gamma]
                tempev[index] = like
            evidence = np.sum(tempev)
            #print(evidence)

            acc_gamma = [None] * len(data[index])
            for index in range(N):
                # collect counts
                for i in range(depth):
                    for j in range(int(post_path[i,-1])):
                        if index == 0:
                            acc_a[i][j] = Ts[index][0][i][j]
                            acc_pi[i][j] = Ts[index][1][i][j]
                            acc_exit[i][j] = Ts[index][2][i][j]
                        else:
                            acc_a[i][j] += Ts[index][0][i][j]
                            acc_pi[i][j] += Ts[index][1][i][j]
                            acc_exit[i][j] += Ts[index][2][i][j]
                # collect production occupancies
                if index == 0:
                    start = 0
                    end = len(data[index])
                else:
                    start = end
                    end += len(data[index])
                acc_gamma[index] = np.zeros((len(data[index]),prod))
                for j in range(int(post_path[-1,-1])):
                    acc_gamma[index] = acc_gamma[index] + Ts[index][3][j]
                gamma_all[start:end,:] = acc_gamma[index]

            # update markov chain parameters
            for i in range(depth):
                for j in range(int(post_path[i,-1])):
                    tm_exit_alpha,tm_exit_mix,evidence = dirichlet_update(np.array(post_tm_exit_alpha[i][j]),\
                                                                          np.array(prior_tm_exit_alpha[i][j]),\
                                                                          np.concatenate((acc_a[i][j],\
                                                                          np.array([acc_exit[i][j]])),axis=0),\
                                                                          evidence)
                    pi_alpha,pi_mix,evidence = dirichlet_update(np.array(post_pi_alpha[i][j]),\
                                                                np.array(prior_pi_alpha[i][j]),acc_pi[i][j],\
                                                                evidence)
                    post_tm[i][j] = tm_exit_mix[0:prior_tm[i][j].shape[0],:]
                    post_exit[i][j] = tm_exit_mix[-1,:]
                    post_pi[i][j] = pi_mix.T

            # finite difference inversion; gradient gaussian used for lowerbound calculation
            for i in range(prod):
                post_mu, post_beta, post_a, post_b, evidence = gauss_update(fulldata,post_a,post_b,post_beta,
                                                                            post_mu,prior_a,prior_b,prior_beta,
                                                                            prior_mu,prior_mixture_mix,
                                                                            gamma_all[:,i].T, i, evidence)

            post_mixture_alpha,post_mixture_mix = dirichlet_update(np.array(post_mixture_alpha),\
                                                                   np.array(prior_mixture_alpha), \
                                                                   gamma_all,evidence)[:-1]

            record_exit.append(post_exit)
            record_pi.append(post_pi)
            record_tm.append(post_tm)
            record_evidence.append(evidence)

            count += 1

            # convergence conditions
            #if count % 5 ==0:
            if count > 2:
                diagnostic = np.abs(np.diff(record_evidence))
                #print(diagnostic)
                diagnostic = diagnostic[-1] / np.abs(record_evidence[-1]);
                if diagnostic < tol or count > max_iter:
                    converged = True

    pathstate = [None] * len(data)
    ideals = [None] * len(data)

    return post_mu, post_beta, post_a, post_b, post_pi, post_tm, post_exit, record_evidence[-1]
    '''
    for index in range(N):
        temp = data[index]
        z, meanpath = viterbi_activation(post_depth, post_path, post_depth_num, post_pi, post_tm, post_mu,
                                         post_exit, Ts[index][3], len(temp))
        pathstate[index] = z
        ideals[index] = meanpath

    return ideals
    '''

