import numpy as np
import pandas as pd

class Optimizer:
    def __init__(self, model, eps) -> None:
        self.m, self.eps = model, eps
        self.trace_obj, self.trace_x = [], []
        self.trace_step, self.trace_d = [], []
    
    def set_init(self, lb = 0, ub = 1)-> None:
        assert(lb <= ub), print("ub is larger than lb")
        self.x = (ub - lb) + np.random.rand(self.m.dim) + lb * np.ones(self.m.dim)
    
    def first_order(self):
        self.grad, self.obj_values = self.m.diff_eval(self.x), self.m.obj_fun_eval(self.x)
        return self.grad, self.obj_values
    
    def second_order(self):
        self.hessian = self.m.diff_second_eval(self.x)
        return self.hessian
    
    def get_step(self):
        # self.step = 0.01
        self.step = self.__armijo_step()
    
    def __armijo_step(self):
        alpha, beta, pho = 10, 0.1, 0.8
        for _ in range(100):
            if self.m.obj_fun_eval(self.x + alpha * self.d) <= self.obj_values + alpha * pho * np.dot(self.grad.T, self.d):
                return alpha
            alpha *= beta
        return False

    def __exact_step(self):
        pass

    def moniter(self, iter_time):
        self.trace_obj.append(self.obj_values)
        self.trace_x.append(self.x)
        self.trace_d.append(self.d)
        self.trace_step.append(self.step)
        if iter_time % 10 == 9:
            print("n iterations {} - objective function: {}".format(iter_time, self.m.obj_fun_eval(self.x)))
    
    def is_stop(self):
        if np.linalg.norm(self.grad, 1.0) <= self.eps:
            return True
        return False
    
    def update(self):
        assert(self.step != False), print("objective function is not decreased")
        self.x += self.step * self.d
    
    def save(self):
        self.file_route = "./" + "results//" 
        pd.DataFrame(self.trace_obj).to_csv(self.file_route +  "results_obj.csv")
        pd.DataFrame(self.trace_x).to_csv(self.file_route + "results_x.csv")
        pd.DataFrame(self.trace_d).to_csv(self.file_route + "results_d.csv")
        pd.DataFrame(self.trace_step).to_csv(self.file_route + "results_step.csv")

class Netwon(Optimizer):
    def __init__(self, model, eps) -> None:
        super().__init__(model, eps)
    
    def search_direction(self):
        self.grad, self.obj_values = self.first_order()
        self.hessian = self.second_order()
        self.d = -1.0 * np.dot(np.linalg.inv(self.hessian), self.grad)

class Gradient(Optimizer):
    def __init__(self, model, eps) -> None:
        super().__init__(model, eps)
    
    def search_direction(self):
        self.grad, self.obj_values = self.first_order()
        self.d = -1.0 * self.grad