import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

input_path = "..."                  #Place your directory here
output_path = "...\\Normalized\\"   #Place your directory here

tons = ['Intact','Damage1', 'Damage2', 'Damage3'] #, 

ton_title_mapping = {
    'Intact':  'PD0',
    'Damage1': 'PD1',
    'Damage2': 'PD2',
    'Damage3': 'PD3',
}

#%% Define the normalize function
def process_caisson_data(filename, r1, r2):
    path = "F:\\PNL\\4. Project\\16. Caisson\\2. Comsol\\JCSHM_review\\"
    df = pd.read_excel(os.path.join(path, filename))
    signal = df.iloc[:, 0]
    ff = np.linspace(0, len(signal) - 1, len(signal))
    ff_s = np.linspace(0, len(signal) - 1, (len(signal) - 1) * 10 + 1)

    f_interp_s = interp1d(ff, signal, kind='linear')
    imp_s = f_interp_s(ff_s)

    signal1 = imp_s[r1 * 10:r2 * 10]
    pks = np.max(signal1)
    locs = np.argmax(signal1)

    for i in range(locs, -1, -1):
        if signal1[i] <= pks * 1 / np.sqrt(2):
            loc_flfu_1 = i
            break
    for i in range(locs, len(signal1)):
        if signal1[i] <= pks * 1 / np.sqrt(2):
            loc_flfu_2 = i
            break

    fl = loc_flfu_1 / 10 + r1
    fu = loc_flfu_2 / 10 + r1
    fp = (locs / 10 + r1)
    damp = np.sqrt(1 / 2 - np.sqrt((4 + 4 * ((fu - fl) / fp) ** 2 - ((fu - fl) / fp) ** 4) ** (-1)))
    ff_x = ff_s / (locs / 10 + r1)
    ff_x1 = np.arange(0.89, 1.11, 0.001)

    # Get the indices of the 50 smallest elements # Get the values of the 50 smallest elements
    smallest_indices = np.argsort(signal1)[:50]
    smallest_value = np.mean(signal1[smallest_indices])

    f_interp = interp1d(ff_x, imp_s, kind='linear',fill_value='extrapolate')
    imp_x = f_interp(ff_x1)
    
    plt.figure()
    plt.semilogy(ff, signal, label='Intact',color='b')
    plt.legend()
    plt.show()   
    return fu,fl,fp,damp,pks,smallest_value

#%% Start the importing loop
for peak in [1,2,3]:
    print(peak)
    if peak == 1:
        r1,r2 =20,60 #Peak 1
    elif peak == 2:
        r1,r2 =150,250 #Peak 2
    elif peak == 3:
        r1,r2 =300,400 #Peak 3
    
    assembles = list(range(1,16))
  
    # Call the function for each caisson
    caisson_files = ['SVD_15ACCs_Caisson1_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson2_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson3_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson4_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson5_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson6_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson7_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson8_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson9_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson10_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson11_Intact_1.xlsx',
                     'SVD_15ACCs_Caisson12_Intact_1.xlsx',
                     ] #'SVD_15ACCs_Caisson3_Intact_1.xlsx'
    count=0
    noc=len(caisson_files) #Number of caisson
    fu,fl,fp_0,damp,pks,smallest_value=np.zeros(noc),np.zeros(noc),np.zeros(noc),np.zeros(noc),np.zeros(noc),np.zeros(noc)
    
    for file in caisson_files:
        fu[count], fl[count], fp_0[count], damp[count],pks[count],smallest_value[count]=process_caisson_data(file, r1, r2)
        count +=1
    print(damp)
      
    #%% Load data
    all_signals = []  # List to store all signals for different tons
    for ton in tons:
        for caisson in range(count):
            signals_for_ton = []  # List to store signals for the current ton       
            df = pd.read_excel(os.path.join(input_path, 'SVD_15ACCs_Caisson{}_{}_1.xlsx'.format(caisson+1,ton)))          
            signal = df.iloc[:, 0]  # Assuming the first column is at index 0
            ff=np.linspace(0,len(signal)-1,len(signal))
            
            ff_x = ff / fp_0[caisson]
            # Normalize ff_x
            ff_x = (ff_x-1) * damp[caisson] / damp[1]+1
    
            # Define the new range for interpolation
            ff_x1 = np.arange(0.9, 1.1, 0.001)
    
            # Interpolate imp_s to imp_x
            f_interp = interp1d(ff_x, signal, kind='linear',fill_value='extrapolate')
            imp_x = f_interp(ff_x1)
    
            #%% Normalize to log scale
            pks_intact,smallest_value_intact=pks[0],smallest_value[0]    
            imp_x=(imp_x-smallest_value_intact)/(pks_intact-smallest_value_intact)
            log_freq_mag = np.log10(imp_x);
    
            signals_for_ton.append((ff_x1, imp_x))
            
            df_with_noise = pd.DataFrame(imp_x)
            df_with_noise.to_excel(os.path.join(output_path,'SVD_15ACCs_Caisson{}_{}_peak_{}.xlsx'.format(caisson+1,ton,peak)), index=False)
        
        #%% Store signals for the current ton in the list of all signals
        all_signals.append((ton, signals_for_ton))

    # Plot all signals together
    for ton, signals_for_ton in all_signals:
        plt.figure(figsize=(3, 2), dpi=600)  # Adjust the size and dpi of each subplot
        for ff_x1, imp_x in signals_for_ton:
            plt.plot(ff_x1, imp_x, label=ton, linewidth=0.15)  # Adjust the line width
            
        # Add vertical dotted lines at x=0.5 and x=1.5
        plt.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.25)
        plt.axvline(x=1.2, color='gray', linestyle='--', linewidth=0.25)    
            
        plt.title('{} of Cassion {:.0f}'.format(ton_title_mapping.get(ton, ton),caisson+1), fontsize=6)  # Set the title font size
        plt.xticks(fontsize=6)  # Set the x-axis tick font size
        plt.yticks(fontsize=6)  # Set the y-axis tick font size
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
        plt.tight_layout()  # Adjust layout to prevent overlap of subplots
        plt.show()
    
    if peak == 1:
        all_signals_1=all_signals
    elif peak == 2:
        all_signals_2=all_signals
    elif peak == 3:
        all_signals_3=all_signals

for j in range(len(tons)):
    # Create subplots with 1 row and 3 columns with no horizontal space between subplots
    fig, axs = plt.subplots(1, 3, figsize=(3, 2), dpi=600, sharey=True, sharex=True, gridspec_kw={'wspace': 0})
    
    # Plot each set of signals on its respective subplot
    for i, all_signals in enumerate([all_signals_1, all_signals_2, all_signals_3]):
        signals_for_ton=all_signals[j][1]
        for ff_x1, imp_x in signals_for_ton:
            axs[i].plot(ff_x1, imp_x, label=ton, linewidth=1.5)
            
      
        # Set x-axis limits for each subplot
        axs[i].set_xlim(0.8, 1.2)
        
        # Set title for each subplot with font size 6
        axs[i].set_title(f'Mode {i+1}', fontsize=6)
        
        # Set tick font size
        axs[i].tick_params(axis='both', which='major', labelsize=6)
        
        # Remove y ticks from the second and third subplots
        axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)
        axs[2].tick_params(axis='y', which='both', left=False, labelleft=False)

    fig.text(0.5, 0.04, '{} of Cassion {:.0f}'.format(ton_title_mapping.get(tons[j], tons[j]), caisson), ha='center', va='center', fontsize=6)

    # Adjust layout
    plt.tight_layout()
    
    plt.show()

