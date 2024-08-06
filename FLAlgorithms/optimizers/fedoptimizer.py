from torch.optim import Optimizer
# 这段代码定义了四个自定义的优化器，分别为 MySGD、FEDLOptimizer、pFedMeOptimizer 和 APFLOptimizer。
# 这些优化器类继承了 torch.optim.Optimizer 并实现了 step 方法，用于更新模型参数。下面是每个类的详细解释：


class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        # __init__ 方法：初始化优化器，设置学习率 lr。
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        # step 方法：执行参数更新。根据是否设置了 beta 值，使用 beta 或学习率 lr 更新参数。
        # 如果 beta 不为 0，使用 beta 乘以梯度来更新参数；否则，使用学习率 lr 进行更新。
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(-beta, d_p)
                else:     
                    p.data.add_(-group['lr'], d_p)
        return loss


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        # __init__ 方法：初始化优化器，设置学习率 lr、服务器梯度 server_grads、之前的梯度 pre_grads 和参数 eta。
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        # step 方法：执行参数更新，使用当前梯度、服务器梯度、之前的梯度和学习率 lr。参数更新公式为：
        # p.data = p.data - \text{lr} \times (p.grad.data + \eta \times \text{server_grads}[i] - \text{pre_grads}[i])
        # 其中，i 用于索引不同的参数。
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data = p.data - group['lr'] * \
                         (p.grad.data + group['eta'] * self.server_grads[i] - self.pre_grads[i])
                # p.data.add_(-group['lr'], p.grad.data)
                i += 1
        return loss


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        # __init__ 方法：初始化优化器，设置学习率 lr、lambda 和 mu。
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        # step 方法：执行参数更新，使用当前梯度、lambda 和 mu。参数更新公式为：
        # p.data=p.data−lr×(p.grad.data+λ×(p.data−localweight.data)+μ×p.da
        # 其中，local_weight_updated 是本地更新的权重。
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        # update_param 方法：将模型参数更新为本地权重 local_weight_updated。
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        # __init__ 方法：初始化优化器，设置学习率 lr。
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, beta = 1, n_k = 1):
        # step 方法：执行参数更新。使用 beta 和 n_k 计算更新量
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta  * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss
