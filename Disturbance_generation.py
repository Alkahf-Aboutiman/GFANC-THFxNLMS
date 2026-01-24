import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import repeat 
from scipy import signal, misc
import torch
import torchaudio

#-------------------------------------------------------------
# Function: Disturbance_reference_generation()
# Description: Generate the disturbance and reference by using the defuat paramters.
#-------------------------------------------------------------
def Disturbance_reference_generation():
    # Defined the configuration for the ANC system
    fs = 16000 # The system sampling rate 
    T = 5 # The duraiton of the simulation 
    t = np.arange(0,T,1/fs).reshape(-1,1)
    f0 = 500
    
    # Constructing the refererence signal
    Re = np.random.randn(len(t))
    
    # Define the low-pass filter
    f_cutoff = 2000 
    N_fc = f_cutoff/fs 
    b1, b2 = signal.firwin(128, N_fc), signal.firwin(128, 2*N_fc)
    
    # Constructing primary path
    Pri_path = signal.convolve(b1,b2)
    w1, h1 = signal.freqz(Pri_path)
    w2, h2 = signal.freqz(b2)

    # Drawing the frequency response of the low-pass filter 
    plt.title('Digital filter frequency response')
    plt.plot(w1, 20*np.log10(np.abs(h1)),'b')
    plt.plot(w2, 20*np.log10(np.abs(h2)),'r')
    plt.ylabel('Amplitude Response (dB)')
    plt.xlabel('Frequency (rad/sample)')
    plt.grid()
    plt.show()

    # Drawing the impulse response of the primary path
    plt.title('The response of the primary path')
    plt.plot(b1)
    plt.ylabel('Amplitude')
    plt.xlabel('Length (taps)')
    plt.grid()
    plt.show()
    
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, Re), signal.lfilter(b2, 1, Re)
    print(Fx[1])
    print(Dir.shape, Fx.shape)

    # Drawing the frequency spectrum of the disturbance 
    f, Pper_spec = signal.periodogram(Dir, fs, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pper_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()

    # Drawing the frequency spectrum of the Fx
    f, Pper_spec = signal.periodogram(Fx, fs, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pper_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)

#-------------------------------------------------------------
# Function: Disturbance_reference_generation_from_Afilter
# Discription : Generating the distubrane and reference signal from the defined parameters
#-------------------------------------------------------------
def Disturbance_reference_generation_from_Afilter(fs, T, f_vector, Pri_path, Sec_path):
    # Pri_path and Sec_path are One dimension arraies
    # f_vector: the filter coefficients in time domain
    
    len_f = 1024
    t = np.arange(0,T,1/fs).reshape(-1,1)
    xin = np.random.randn(len(t))
    Re = signal.lfilter(f_vector, 1, xin) # random signal pass the bandpass filter to get the noise
    Noise = Re[len_f-1:]
    
    # Construting the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)


def Disturbance_reference_generation_from_Afilter_fs_3000(fs_in, fs_out, T, f_vector, Pri_path, Sec_path):
    """
    Génère un signal de référence bruité, le filtre, et rééchantillonne le tout.

    Paramètres :
        fs_in     : fréquence d'échantillonnage initiale
        fs_out    : fréquence d'échantillonnage de sortie
        T         : durée du signal (en secondes)
        f_vector  : coefficients d’un filtre pour générer le bruit
        Pri_path  : filtre primaire
        Sec_path  : filtre secondaire
    Retourne :
        Dir_out, Fx_out : signaux filtrés et rééchantillonnés (tensors PyTorch)
    """
    len_f = 1024
    t = np.arange(0, T, 1 / fs_in).reshape(-1, 1)
    xin = np.random.randn(len(t)).flatten()

    # Filtrage initial pour générer du bruit
    Re = signal.lfilter(f_vector, 1, xin)
    Noise = Re[len_f - 1:]

    # Convolution avec les chemins primaire et secondaire
    Dir = signal.lfilter(Pri_path, 1, Noise)
    Fx = signal.lfilter(Sec_path, 1, Noise)

    # Convertir en tensors PyTorch pour resampling
    Dir_tensor = torch.from_numpy(Dir).unsqueeze(0).float()
    Fx_tensor = torch.from_numpy(Fx).unsqueeze(0).float()
    Noise_tensor = torch.from_numpy(Noise).unsqueeze(0).float()

    # Rééchantillonnage
    Dir_resampled = torchaudio.functional.resample(Dir_tensor, orig_freq=fs_in, new_freq=fs_out)
    Fx_resampled = torchaudio.functional.resample(Fx_tensor, orig_freq=fs_in, new_freq=fs_out)

    return Dir_resampled.squeeze(0), Fx_resampled.squeeze(0)

def Disturbance_reference_generation_from_Afilter_LTV(fs, T, f_vector, Pri_path, Sec_path):
    # Pri_path and Sec_path are One dimension arraies
    # f_vector: the filter coefficients in time domain
    
    len_f = 1024
    t = np.arange(0,T,1/fs).reshape(-1,1)
    xin = np.random.randn(len(t))
    Re = signal.lfilter(f_vector, 1, xin) # random signal pass the bandpass filter to get the noise
    Noise = Re[len_f-1:]
 
    # Calculate the signal power
    signal_power = np.mean(Noise**2)
    
    # Split the signal into three parts
    N = len(Noise)
    third_N = N // 3
    
    # Define SNR levels for the three segments
    SNR_start = float('inf')  # No modification to the first segment
    SNR_middle = 5            # SNR of 5 dB for the second segment
    SNR_end = 8          # SNR of 15 dB for the third segment
    
    # Calculate noise power for the middle and end segments
    noise_middle_power = signal_power / (10**(SNR_middle / 10))
    noise_end_power = signal_power / (10**(SNR_end / 10))
    
    # Leave the first segment unchanged
    noise_start = Noise[:third_N]
    
    # Adjust noise power for the middle and end segments
    noise_middle = Noise[third_N:2*third_N] * np.sqrt(noise_middle_power / np.mean(Noise[third_N:2*third_N]**2))
    noise_end = Noise[2*third_N:] * np.sqrt(noise_end_power / np.mean(Noise[2*third_N:]**2))
    
    # Combine all segments
    wavec_SNR = np.concatenate((noise_start, noise_middle, noise_end), axis=0)
    
    # Construting the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec_SNR), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float)

#-------------------------------------------------------------
# Function: Disturbance_generation_from_real_noise()
# Descritption: Generating the disturbance and filtered reference from the raw waveform
#-------------------------------------------------------------
def Disturbance_generation_from_real_noise(fs, Repet, wave_form, Pri_path, Sec_path):
    wave = wave_form[0,:].numpy()
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec,wave),axis=0) # add the length of the wave_form through repetition
    pass

    # Construting the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec), signal.lfilter(Sec_path, 1, wavec)
    
    N = len(Dir)
    N_z = N//fs
    Dir, Fx = Dir[0:N_z*fs], Fx[0:N_z*fs]
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(wavec).type(torch.float)


def Disturbance_generation_from_real_noise_3000_fs(fs_orig, fs_in, fs_out, Repet, wave_form, Pri_path, Sec_path):
    # Convertir en numpy et répéter le signal
    wave = wave_form[0, :].numpy()
    wavec = np.tile(wave, Repet + 1)  # +1 pour inclure la version d'origine aussi

    # Resampler le signal à fs_in
    wavec_tensor = torch.from_numpy(wavec).unsqueeze(0).float()
    wavec_resampled = torchaudio.functional.resample(wavec_tensor, orig_freq=fs_orig, new_freq=fs_in)
    wavec_resampled_np = wavec_resampled[0].numpy()

    # Appliquer les filtres (convolution)
    Dir = signal.lfilter(Pri_path, 1, wavec_resampled_np)
    Fx = signal.lfilter(Sec_path, 1, wavec_resampled_np)

    # Resampler les signaux Dir et Fx à fs_out
    Dir_tensor = torch.from_numpy(Dir).unsqueeze(0).float()
    Fx_tensor = torch.from_numpy(Fx).unsqueeze(0).float()
    wavec_tensor = torch.from_numpy(wavec_resampled_np).unsqueeze(0).float()
    Dir_resampled = torchaudio.functional.resample(Dir_tensor, orig_freq=fs_in, new_freq=fs_out)
    Fx_resampled = torchaudio.functional.resample(Fx_tensor, orig_freq=fs_in, new_freq=fs_out)
    wavec_resampled = torchaudio.functional.resample(wavec_tensor, orig_freq=fs_in, new_freq=fs_out)

    # Retourner les signaux
    return Dir_resampled[0], Fx_resampled[0], wavec_resampled[0]
#-------------------------------------------------------------
# Function: Disturbance_generation_from_real_noise()
# Descritption: Generating the disturbance and filtered reference from the raw waveform
#-------------------------------------------------------------
def Disturbance_generation_from_real_noise_LTV(fs, Repet, wave_form, Pri_path, Sec_path):
    wave = wave_form[0,:].numpy()
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec,wave),axis=0) # add the length of the wave_form through repetition
    pass
    
    # Calculate the signal power
    signal_power = np.mean(wavec**2)
    
    # Split the signal into three parts
    N = len(wavec)
    third_N = N // 3
    
    # Define SNR levels for the three segments
    SNR_start = float('inf')  # No modification to the first segment
    SNR_middle = 5            # SNR of 5 dB for the second segment
    SNR_end = 8          # SNR of 15 dB for the third segment
    
    # Calculate noise power for the middle and end segments
    noise_middle_power = signal_power / (10**(SNR_middle / 10))
    noise_end_power = signal_power / (10**(SNR_end / 10))
    
    # Leave the first segment unchanged
    noise_start = wavec[:third_N]
    
    # Adjust noise power for the middle and end segments
    noise_middle = wavec[third_N:2*third_N] * np.sqrt(noise_middle_power / np.mean(wavec[third_N:2*third_N]**2))
    noise_end = wavec[2*third_N:] * np.sqrt(noise_end_power / np.mean(wavec[2*third_N:]**2))
    
    # Combine all segments
    wavec_SNR = np.concatenate((noise_start, noise_middle, noise_end), axis=0)

    # Construting the desired signal
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec_SNR), signal.lfilter(Sec_path, 1, wavec)
    
    N = len(Dir)
    N_z = N//fs
    Dir, Fx = Dir[0:N_z*fs], Fx[0:N_z*fs]
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(wavec).type(torch.float)

#-------------------------------------------------------------
# Function: Varied_distrubance_reference_generation_from_Fvector()
# Discription : Generating the distubrane and reference signal from the defined parameters
#-------------------------------------------------------------
def Varied_distrubance_reference_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    t = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    for ii in range(len(f_vector)):
        b2 = signal.firwin(len_f, f_vector[ii], pass_zero='bandpass', window ='hamming',fs=fs)
        xin = np.random.randn(len(t))
        Re = signal.lfilter(b2,1,xin)
        if ii == 0:
            Noise = Re[fs:]
        else:
            if ii ==2 :
                Noise = np.concatenate((Noise, 4*Re[fs:]),axis=0)
            else: 
                Noise = np.concatenate((Noise, Re[fs:]),axis=0)
        
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(Noise).type(torch.float)