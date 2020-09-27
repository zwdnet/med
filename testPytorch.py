# coding:utf-8
# pytorch测试


# 用numpy实现神经网络
def WU():
    import numpy as np
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)
    
    learning_rate = 1e-6
    for t in range(500):
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)
        
        loss = np.square(y_pred - y).sum()
        print(t, loss)
        
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
        
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    print(y - y_pred)
    
    
# 张量版的神经网络
def tensor_nn():
    import torch
    dtype = torch.float
    device = torch.device("cpu")
    
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)
    
    learning_rate = 1e-6
    for t in range(500):
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)
        
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)
            
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)
        
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        
        
# 自动梯度
def AG():
    import torch
    
    dtype = torch.float
    device = torch.device("cpu")
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)        
           
    learning_rate = 1e-6
    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss)
        loss.backward()
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()
            
            
# 自己定义前向后向过程
def DefFB():
    import torch
    
    class MyReLU(torch.autograd.Function):
        
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(min=0)
            
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return grad_input
            
            
    dtype = torch.float
    device = torch.device("cpu")
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)        
           
    learning_rate = 1e-6
    for t in range(500):
        relu = MyReLU.apply
        
        y_pred = relu(x.mm(w1)).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss)
        loss.backward()
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()
            
        
if __name__ == "__main__":
    WU()
    tensor_nn()
    AG()
    DefFB()
