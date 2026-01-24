import torch
import numpy as np
import torch.optim as optim
import progressbar
import scipy.signal as signal


def f_sef(y_batch, eta, num_points=1000):
    eta = torch.tensor(eta, dtype=torch.float32, requires_grad=False).to(y_batch.device)
    z = torch.linspace(0, 1, num_points, dtype=torch.float32, device=y_batch.device)
    z = z.view(1, -1) * y_batch.unsqueeze(-1)
    integrand = torch.exp(-z ** 2 / (2 * eta ** 2))
    integral = torch.trapz(integrand, z, dim=-1)
    return integral


# ------------------------------------------------------------------------------
# Class: THFxNLMS algorithm (nonlinear FxNLMS variant)
# ------------------------------------------------------------------------------
class THFxNLMS():

    def __init__(self, Len, eta, rho=1.0, lambda_factor=0.95):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.eta = eta  # nonlinearity strength
        self.rho = rho  # control parameter for b
        self.lambda_factor = lambda_factor  # forgetting factor
        self.Pe = 0  # energy estimation initialization

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf
        yt = self.Wc @ self.Xd.t()

        power = self.Xd @ self.Xd.t()  # power modified version used for FxNLMS
        return yt, power

    def LossFunction(self, y, d, power):
        e = d - y  # disturbance - control signal
        with torch.no_grad():  # disable gradient tracking here
            # Exponential moving average for error energy estimation
            self.Pe = self.lambda_factor * self.Pe + (1 - self.lambda_factor) * e.pow(2).mean()

        b = self.rho / (1 + torch.exp(-self.Pe))  # dynamic b adaptation
        epsilon = 2.2204e-16
        loss = (torch.tanh(b * e) ** 2) / (2 * power + epsilon)
        return loss, e

    def _get_coeff_(self):
        return self.Wc.detach().numpy()


# ------------------------------------------------------------------------------
# Function : THFxNLMS training loop
# ------------------------------------------------------------------------------
def train_thfxnlms_algorithm(Model, Ref, Disturbance, Stepsize=0.0001):
    bar = progressbar.ProgressBar(maxval=2 * Disturbance.shape[0], \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    optimizer = optim.SGD([Model.Wc], lr=Stepsize)

    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    for itera in range(len_data):
        # Feedforward Fx
        xin = Ref[itera]
        dis = Disturbance[itera]
        y, power = Model.feedforward(xin)

        # Apply nonlinear SEF function on y
        y_non_linear = f_sef(y, Model.eta)

        # Compute loss using nonlinear output
        loss, e = Model.LossFunction(y_non_linear, dis, power)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())

        # Progress bar update
        bar.update(2 * itera + 2)
    bar.finish()
    return Erro_signal


# ------------------------------------------------------------
# Function : Generate broadband noise test waveform
# ------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window='hamming', fs=fs)
    N = filter_len + Seconds * fs
    xin = np.random.randn(N)
    y = signal.lfilter(bandpass_filter, 1, xin)
    yout = y[filter_len:]
    # Standardize
    yout = yout / np.sqrt(np.var(yout))
    # return tensor of shape [1 x sample_rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)
