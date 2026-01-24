# Fixed filter noise cancellation by Sample
import numpy as np
import torch
import torch.optim as optim


def f_sef(y_batch, eta, num_points=1000):
    eta = torch.tensor(eta, dtype=torch.float32, requires_grad=False).to(y_batch.device)
    z = torch.linspace(0, 1, num_points, dtype=torch.float32, device=y_batch.device)
    z = z.view(1, -1) * y_batch.unsqueeze(-1)
    integrand = torch.exp(-z ** 2 / (2 * eta ** 2))
    integral = torch.trapz(integrand, z, dim=-1)
    return integral


# ------------------------------------------------------------------------------
# Class: THFxNLMS algorithm
# ------------------------------------------------------------------------------
class THFxNLMS():

    def __init__(self, Len, eta, Ws, rho=1.0, lambda_factor=0.95):
        self.Wc = torch.tensor(Ws, requires_grad=True)  # Ws: initial coefficients determined by SFANC
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.eta = eta  # Nonlinearity strength
        self.rho = rho  # Control parameter for b
        self.lambda_factor = lambda_factor  # Forgetting factor
        self.Pe = 0  # Initialization of energy estimation

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf
        yt = self.Wc @ self.Xd.t()

        power = self.Xd @ self.Xd.t()  # FxNLMS is different from FxLMS
        return yt, power

    def LossFunction(self, y, d, power):
        e = d - y  # disturbance-control signal
        with torch.no_grad():  # Disable gradient tracking for this update
            self.Pe = self.lambda_factor * self.Pe + (1 - self.lambda_factor) * e.pow(2).mean()

        b = self.rho / (1 + torch.exp(-self.Pe))  # Dynamic calculation of b
        epsilon = 2.2204e-16
        loss = (torch.tanh(b * e) ** 2) / (2 * power + epsilon)
        return loss, e

    def _get_coeff_(self):
        return self.Wc.detach().numpy()


class GFANC_THFxNLMS():
    def __init__(self, Filter_vector, fs, eta):
        self.Filter_vector = torch.from_numpy(Filter_vector).type(torch.float)
        Len = self.Filter_vector.shape[1]
        self.fs = fs
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
        self.eta = eta

    def noise_cancellation(self, Dis, Fx, filter_index, Stepsize, filter_len):
        Error = []
        j = 0
        model = THFxNLMS(Len=filter_len, Ws=self.Current_Filter, eta=self.eta)
        optimizer = optim.SGD([model.Wc], lr=Stepsize)  # Stepsize = learning_rate

        for ii, dis in enumerate(Dis):
            y, power = model.feedforward(Fx[ii])
            y_non_linear = f_sef(y, self.eta)  # Apply SEF function on y
            loss, e = model.LossFunction(y_non_linear, dis, power)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Error.append(e.item())

            if ii == self.fs - 1:  # At the end of the first second
                print(j)
                if not self.Current_Filter[0].equal(self.Filter_vector[j]):
                    print('change the initial weights of THFxNLMS')
                    self.Current_Filter = self.Filter_vector[j].unsqueeze(0)
                    model = THFxNLMS(Len=filter_len, Ws=self.Current_Filter, eta=self.eta)
                    optimizer = optim.SGD([model.Wc], lr=Stepsize)

        return Error
