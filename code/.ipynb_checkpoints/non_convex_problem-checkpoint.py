from abc import ABC, abstractmethod

class NonConvexProblem(ABC):

    @abstractmethod
    def construct_A_matrix(self):
        pass

    @abstractmethod
    def construct_Ai_matrix(self, i):
        pass

    @abstractmethod
    def f_i(self, i, x):
        pass

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
