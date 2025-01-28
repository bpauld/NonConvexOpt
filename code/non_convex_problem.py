from abc import ABC, abstractmethod
import numpy as np

class NonConvexProblem(ABC):

    @abstractmethod
    def get_di(self, i):
        pass

    #allows to handle the representation of y_k internally
    #if di is the same for all i, then most likely the most efficient is to keep y_k as a numpy array of shape (di x n)
    #if not, then probably one should set y_k to be a dicitionnary of size n where y_k[i] is a 1-dimensional array of size di
    @abstractmethod
    def get_y_ik(self, i, y_k):
        pass

    #@abstractmethod
    #def construct_A_matrix(self):
    #    pass

    #@abstractmethod
    #def construct_Ai_matrix(self, i):
    #    pass

    @abstractmethod
    def compute_Ai_dot_y(self, i, y):
        pass

    @abstractmethod
    def f_i(self, i, x):
        pass

    # computes f_i at every column of x
    # worth re-implementing for some problems to be more efficient than a for loop over the columns of x
    def f_i_vec(self, i, x):
        fi = np.zeros(x.shape[1])
        for k in range(x.shape[1]):
            fi[k] = self.f_i(i, x[:, k])
        return fi

    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def f_biconjugate_i(self, i, x): #maybe this is not needed
        pass

    @abstractmethod
    def f_conjugate_i(self, i, z):
        pass
        
    @abstractmethod
    def compute_linear_min_dom_fi(self, i, z):
        pass

    @abstractmethod
    def lmo_1(self, i, alpha, g):
        pass

    @abstractmethod
    def get_feasible_point(self):
        pass

