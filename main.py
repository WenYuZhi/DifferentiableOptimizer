from model import Model
from optimizer import Gradient
from optimizer import Netwon

dim = 10
model = Model(dim = dim)
model.diff()
model.diff_second()

n_iter, eps = 100, 10**(-6)
optimizer = Gradient(model, eps)
optimizer.set_init(lb=0, ub=100)

for i in range(n_iter):
    optimizer.search_direction()
    optimizer.get_step()
    optimizer.update()
    optimizer.moniter(iter_time = i)
    if optimizer.is_stop():
        break

optimizer.save()

