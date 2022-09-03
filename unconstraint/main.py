from model import Model
from optimizer import Gradient
from optimizer import Netwon
from optimizer import ModifiedNetwon
from optimizer import BFGS

dim = 10
model = Model(dim = dim)
model.set_objective(name = 'quad_fun1')
model.diff()
model.diff_second()

n_iter, eps = 100, 10**(-6)
optimizer = ModifiedNetwon(model, eps)
optimizer.set_init(lb=0, ub=100)

for k in range(n_iter):
    optimizer.search_direction(miu = 0.1)
    optimizer.get_step()
    optimizer.update()
    optimizer.moniter(iter_time = k)
    if optimizer.is_stop():
        break

# optimizer.save()

