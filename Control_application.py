import torch
import numpy as np
import matplotlib.pyplot as plt

from loading_real_wave_noise import loading_real_wave_noise
from Reading_path_test import loading_paths_from_MAT
from Disturbance_generation import Disturbance_generation_from_real_noise_3000_fs
from THFxNLMS_algorithm import THFxNLMS, train_thfxnlms_algorithm
from Control_filter_selection_soft_index import Control_filter_selection
from GFANC_NL import Fixed_filter_controller
from GFANC_THFxNLMS_once import GFANC_THFxNLMS

# ANC parameters
fs = 3000
step_size = 0.05
eta = 5
filter_len = 256
duration = 5
Wc_F = [20, 500]

# ------------------------------------------------------------
# Load one real noise sample
# ------------------------------------------------------------
sound_name = 'Engine'
waveform, _ = loading_real_wave_noise(
    folde_name='Noises/',
    sound_name=sound_name + '.wav',
    fs=fs
)

# Load Primary / Secondary paths
Pri_path, Sec_path = loading_paths_from_MAT(
    folder='Control_Paths',
    subfolder='Noise_box_with_opening',
    Pri_path_file_name='primary_path.mat',
    Sec_path_file_name='secondary_path.mat'
)

# Disturbance generation (ANC input test)
Dis, Fx, _ = Disturbance_generation_from_real_noise_3000_fs(
    fs_orig=fs, fs_in=3000, fs_out=fs, Repet=0,
    wave_form=waveform, Pri_path=Pri_path, Sec_path=Sec_path
)

Time = np.arange(len(Dis)) * (1 / fs)

# ------------------------------------------------------------
# Baseline: THFxNLMS
# ------------------------------------------------------------
controller = THFxNLMS(Len=filter_len, eta=eta)
Error_THFxNLMS = train_thfxnlms_algorithm(controller, Ref=Fx, Disturbance=Dis, Stepsize=step_size)

plt.figure()
plt.title('THFxNLMS Algorithm')
plt.plot(Time, Dis, label='ANC off', color='blue')
plt.plot(Time, Error_THFxNLMS, label='ANC on', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------------------
# GFANC - Sub-filter selection
# ------------------------------------------------------------
MODEL_PTH = 'models/Pretrained_Neural_Network/Pretrained_CNN.pth'
path_mat = 'models/Control_filter/Pretrained_sub_Control_filters_Noise_Box_exp.mat'

Filter_vector = Control_filter_selection(
    fs=fs,
    MODEL_PTH=MODEL_PTH,
    path_mat=path_mat,
    Primary_noise=torch.tensor(Dis).unsqueeze(0).float()
)

FixedCtrl = Fixed_filter_controller(Filter_vector=Filter_vector, fs=fs, eta=eta)
Error_GFANC = FixedCtrl.noise_cancellation(Dis=Dis, Fx=Fx)

plt.title('GFANC Algorithm')
plt.plot(Time, Dis, label='ANC off', color='blue')
plt.plot(Time, Error_GFANC, label='ANC on', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Hybrid GFANC - THFxNLMS
# ------------------------------------------------------------
HybridGFANC = GFANC_THFxNLMS(Filter_vector=Filter_vector, fs=fs, eta=eta)
Error_Hybrid = HybridGFANC.noise_cancellation(
    Dis=Dis, Fx=Fx, filter_index=Filter_vector, Stepsize=step_size, filter_len=filter_len
)

plt.title('Hybrid GFANC-THFxNLMS Algorithm')
plt.plot(Time, Dis, label='ANC off', color='blue')
plt.plot(Time, Error_Hybrid, label='ANC on', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.show()
