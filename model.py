import sympy
import numpy as np

class Model:
    def __init__(self, dim:int) -> None:
        assert(dim > 0 and type(dim) == int)
        self.dim = dim
        self.x = self.add_vars(self.dim)
        self.obj_func = self.set_objective()
    
    def add_vars(self, dim):
        return [sympy.Symbol('x_{}'.format(i)) for i in range(dim)]
    
    def set_objective(self):
        return self.__quad_fun2()
    
    def __quad_fun(self):
        s = 0
        for i in range(self.dim):
            s += self.x[i]**2
        return s
    
    def __quad_fun1(self):
        s = 0
        for i in range(self.dim):
            s += (2*i+1)*self.x[i]**2
        return s
    
    def __quad_fun2(self):
        s = self.x[-1]**2
        for i in range(self.dim - 1):
            s += (self.x[i] - 10 * self.x[i+1])**2
        return s

    def obj_fun_eval(self, x:list):
        assert(len(x) == self.dim), print(" dimension is error ")
        s = dict(list(zip(self.x, x)))
        return self.obj_func.evalf(subs = s)
    
    def diff(self):
        self.diff_obj = [sympy.diff(self.obj_func, self.x[i], 1) for i in range(self.dim)]
        # print(self.diff_obj)
    
    def diff_eval(self, x:list):
        s = dict(list(zip(self.x, x)))
        grad = np.array([self.diff_obj[i].subs(s) for i in range(self.dim)])
        grad = grad.astype('float')
        return grad
    
    def diff_second(self):
        self.hessian_obj = [[sympy.diff(self.diff_obj[i], self.x[j], 1) for i in range(self.dim)] for j in range(self.dim)]
    
    def diff_second_eval(self, x:list):
        s = dict(list(zip(self.x, x)))
        hessian = np.array([self.hessian_obj[i][j].subs(s) for i in range(self.dim) for j in range(self.dim)]).reshape((self.dim, self.dim))
        hessian = hessian.astype('float')
        return hessian

if __name__ == "__main__":
    dim = 4
    model = Model(dim = dim)
    x = [1 for i in range(dim)]
    print("x: ", x)
    print("obj function value: ", model.obj_fun_eval(x))
    model.diff()
    print("first order value: ", model.diff_eval(x))
    model.diff_second()
    print("second order value: ", model.diff_second_eval(x))