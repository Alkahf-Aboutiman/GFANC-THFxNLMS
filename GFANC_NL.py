# Fixed filter noise cancellation by Sample
import numpy as np
import torch

def f_sef(y_batch, eta, num_points=1000):
    eta = torch.tensor(eta, dtype=torch.float32, requires_grad=False).to(y_batch.device)
    z = torch.linspace(0, 1, num_points, dtype=torch.float32, device=y_batch.device)
    z = z.view(1, -1) * y_batch.unsqueeze(-1)
    integrand = torch.exp(-z**2 / (2 * eta**2))
    integral = torch.trapz(integrand, z, dim=-1)
    return integral


class Fixed_filter_controller():
    def __init__(self, Filter_vector, fs, eta):
        self.Filter_vector = torch.from_numpy(Filter_vector).type(torch.float)# torch.Size([Xseconds, 1024])
        Len = self.Filter_vector.shape[1]
        self.fs = fs
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
        self.eta = eta
    
    def noise_cancellation(self, Dis, Fx):
        Erro = torch.zeros(Dis.shape[0])
        j = 0
        for ii, dis in enumerate(Dis):
            self.Xd = torch.roll(self.Xd,1,1)
            self.Xd[0,0] = Fx[ii] # Fx[ii]: fixed-x signal
            yt = self.Current_Filter @ self.Xd.t()
            y_non_linear = f_sef(yt, self.eta)  # Appliquer la fonction SEF sur y
            e = dis - y_non_linear
            Erro[ii] = e.item()
            if (ii + 1) % self.fs == 0:
                self.Current_Filter = self.Filter_vector[j]
                j += 1
        return Erro