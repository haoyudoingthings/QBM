# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# This notebook aims to recreate an annealer machine running simulated quantum annealing.
# 
# There are two designs present. One employs sequential update on the spins without global moves, and the other updates randomly and incorporates global moves.
# Reference to the first design: [OpenCL-based design of an FPGA accelerator for quantum annealing simulation](https://link.springer.com/article/10.1007%2Fs11227-019-02778-w)
# Reference to teh second design: [Quantum annealing by the path-integral Monte Carlo method: The two-dimensional random Ising model](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.66.094203)
# 

# %%
import numpy as np


# %%
def one_SQA_run(J, h, trans_fld_sched, M, T, init_state=None):
    """
    One simulated quantum annealing run over the full transverse field strength schedule.
    The goal is to find a state such that sum(J[i, i]*state[i]) + sum(J[i, j]*state[i]*state[j]) is minimized.
    
    Parameters:
        J (2-D array of float): The matrix representing the coupling field of the problem.
        h (1-D array of float): The vector representing the local field of the problem.
        trans_fld_sched (list[float]): The transeverse field strength schedule for QA.
                                       The number of iterations is implicitly the length of trans_fld_schedule.
        M (int): Number of Trotter replicas. Larger M leads to higher probability of finding ground state.
        T (float): Temperature parameter. Smaller T leads to higher probability of finding ground state.
        init_state (1-D array of int, default=None): The boolean vector representing the initial state.
                                                     If None, a random state is chosen.
    
    Return: final_state (1-D array of int)
    """
    if np.any(np.diag(J)):
        raise ValueError("Diagonal elements of J should be 0.")

    # J: block sparse matrices with block size of (N, N)
    N = J.shape[0]
    j = 0.5*(J + J.T) # making sure J is symmetric
    j = np.kron(np.eye(M), j/M) # block diagonal of J, repeated M times and divided by M
    
    h_extended = np.repeat(h/M, M)

    Jp_terms = np.eye(N*M, k=N) + np.eye(N*M, k=N*(1-M))
    Jp_terms = 0.5*(Jp_terms + Jp_terms.T)
    
    if init_state is None:
        state = 2 * np.random.binomial(1, 0.5, N*M) - 1
    else:
        state = np.tile(init_state, M)
    
    
    for Gamma in trans_fld_sched:
        Jp_coef = -0.5 * T * np.log(np.tanh(Gamma / M / T))
        
        # First design (Tohoku)
        for flip in range(N*M): # can be parallelized
            delta_E = -4 * (j - Jp_coef * Jp_terms)[flip].dot(state) * state[flip] - 2 * h_extended[flip] * state[flip]
            if np.random.binomial(1, np.minimum(np.exp(-delta_E/T), 1.)):
                state[flip] *= -1

        # # Second design (PRB.66.094203)
        # # Local move
        # flip = np.random.randint(N*M)
        # delta_E = -4 * (J - Jp_coef * Jp_terms)[flip].dot(state) * state[flip] - 2 * h[flip] * state[flip]
        # if np.random.binomial(1, np.minimum(np.exp(-delta_E/T), 1.)):
        #     state[flip] *= -1
        
        # # Global move
        # flip = (np.arange(N*M) % N == np.random.randint(N))
        # delta_E = -4 * J[flip].dot(state).dot(state[flip]) - 2 * h[flip].dot(state[flip])
        # if np.random.binomial(1, np.minimum(np.exp(-delta_E/T), 1.)):
        #     state[flip] *= -1
        
        # state_history.append(state.copy())
    
    return state


# %%
def main():
    import time
    import matplotlib.pyplot as plt


    J = np.array([[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    N = J.shape[0]
    ansatz = np.zeros(J.shape[0], dtype=np.bool_)

    M = 40
    T = 0.05

    steps = 10**4
    Gamma0 = 3
    #schedule = [Gamma0 / (1 + a) for a in range(steps)]
    schedule = np.linspace(Gamma0, 10**(-8), num=steps)


    # state_history = []

    np.random.seed(0)
    start_time = time.time()
    ans = one_SQA_run(J, schedule, M, T)
    total_time = time.time() - start_time
    print(f"ground state: {np.sign(np.sum(np.reshape(ans, (M, N)), axis=0))}")
    print(f'time: {total_time} s')


    # true_percent = []
    # for i in range(N):
    #     true_percent.append([0.5*(np.sum(a[i::N])/M + 1) for a in state_history])


    # fig = plt.figure(dpi=120)
    # x = np.arange(steps)
    # #y = [sum(ans_dict[b].values())/3 for b in sorted(args_lst, key=lambda a: sum(ans_dict[a].values()))]

    # for i in range(N):
    #     plt.plot(x, true_percent[i], label=f"spin {i+1}")
    # plt.legend()


# %%
if __name__ == "__main__":
    main()


# %%
