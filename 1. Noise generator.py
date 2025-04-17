import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_path = "...\\Normalized\\"    #Place your directory here
output_path = "...\\Noised\\"       #Place your directory here
tons = {'Intact','Damage1', 'Damage2', 'Damage3'};

counts = list(range(1,13))
peaks=[1,2,3]

#%% Load data
for ton in tons:
    for count in counts:
        signal = []
        for peak in peaks:
            df = pd.read_excel(os.path.join(input_path, 'SVD_15ACCs_Caisson{}_{}_peak_{}.xlsx'.format(count,ton,peak)))
            signal1 = df.iloc[000:200, 0]  # Assuming the first column is at index 0
            signal = np.concatenate((signal, signal1))
            
        signal_length=len(signal)
        
        #%% Adding noise using percentage
        for noise_percent in [0,1,2,3,4,5,6,8,10,12,14,16,18,20]:  # Assuming you want to iterate from 1% to 10%
            noise_assembles=10
            if noise_percent==0 :
                noise_assembles=1
            for _ in range(noise_assembles):  # Repeat 10 times for each noise level
                noise_amplitude = noise_percent / 100.0 * np.max(signal)
            
            
                # --- Generate Separate Noise Components ---
                # 1. White Gaussian Noise
                white_noise = np.random.normal(0, noise_amplitude, size=signal_length)

                # 2. Impulse Noise (spikes)
                max_val = np.max(np.abs(signal))
                impulse_noise = np.zeros(signal_length)
                impulse_prob = 0.10  # 1% impulses
                num_impulses = int(impulse_prob * signal_length)
                impulse_amplitude = noise_percent / 100.0 * max_val  # scale with noise level
                impulse_positions = np.random.choice(signal_length, num_impulses, replace=False)
                impulse_values = np.random.choice([impulse_amplitude, -impulse_amplitude], num_impulses)
                random_factors = np.random.uniform(0.3, 1, num_impulses)
                impulse_noise[impulse_positions] = impulse_values * random_factors

                # 3. Burst Noise (continuous segment)
                burst_noise = np.zeros(signal_length)
                burst_length = 5  # length of each burst
                num_bursts = 3
                for aa in range(num_bursts):
                    start = np.random.randint(0, signal_length - burst_length)
                    burst_noise[start:start + burst_length] = np.random.normal(0, noise_amplitude * 2, burst_length)

                # 4. Laplace Noise (heavy-tailed)
                laplace_noise = np.random.laplace(0, noise_amplitude / np.sqrt(2), size=signal_length)

                # Combine (choose what to add)
                noisy_signal = signal + white_noise + impulse_noise + 0*burst_noise + 0*laplace_noise
                

                
                df_with_noise = pd.DataFrame(noisy_signal)
                df_with_noise.to_csv(os.path.join(output_path,'SVD_15ACCs_Caisson{}_{}_noise_{}_asb_{}.csv'.format( count,ton, noise_percent,_+1)), index=False)



