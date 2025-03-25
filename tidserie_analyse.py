import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal

# Indlæs datasættene
train_df = pd.read_csv("DailyDelhiClimateTrain.csv")
test_df = pd.read_csv("DailyDelhiClimateTest.csv")

# Konverter dato-kolonnen til datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Kombinér datasættene for at få en fuld tidsserie
full_df = pd.concat([train_df, test_df]).sort_values('date')

# Del 1: Visualisering af originale tidsseriedata
plt.figure(figsize=(14, 8))
plt.plot(full_df['date'], full_df['meantemp'], label='Gennemsnitstemperatur')
plt.title('Daglig Gennemsnitstemperatur i Delhi (2013-2017)', fontsize=16)
plt.xlabel('Dato')
plt.ylabel('Temperatur (°C)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('original_temperatur_serie.png')
plt.show()

# Del 2: Vurdering af støj med forskellige tidsperioder (daglig, ugentlig, månedlig)
# Resample til forskellige tidsperioder
daily_data = full_df.copy()
weekly_data = full_df.set_index('date').resample('W').mean().reset_index()
monthly_data = full_df.set_index('date').resample('M').mean().reset_index()

plt.figure(figsize=(14, 10))

# Daglig data (med støj)
plt.subplot(3, 1, 1)
plt.plot(daily_data['date'], daily_data['meantemp'], 'b-', alpha=0.7)
plt.title('Daglig Gennemsnitstemperatur (Høj støj)')
plt.ylabel('Temperatur (°C)')
plt.grid(True, alpha=0.3)

# Ugentlig data (mindre støj)
plt.subplot(3, 1, 2)
plt.plot(weekly_data['date'], weekly_data['meantemp'], 'g-')
plt.title('Ugentlig Gennemsnitstemperatur (Medium støj)')
plt.ylabel('Temperatur (°C)')
plt.grid(True, alpha=0.3)

# Månedlig data (mindst støj)
plt.subplot(3, 1, 3)
plt.plot(monthly_data['date'], monthly_data['meantemp'], 'r-')
plt.title('Månedlig Gennemsnitstemperatur (Lav støj)')
plt.ylabel('Temperatur (°C)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperatur_resampling.png')
plt.show()

# Del 3: Sammenligning af original data med glidende gennemsnit for at vurdere støj
plt.figure(figsize=(14, 10))

# Forskellige vinduesstørrelser for glidende gennemsnit
window_sizes = [3, 7, 30]
colors = ['g', 'r', 'purple']
labels = ['3-dages glidende gennemsnit', '7-dages glidende gennemsnit', '30-dages glidende gennemsnit']

# Plot original data
plt.plot(full_df['date'], full_df['meantemp'], 'b-', alpha=0.4, label='Original Temperatur')

# Plot glidende gennemsnit med forskellige vinduesstørrelser
for i, window in enumerate(window_sizes):
    rolling_mean = full_df['meantemp'].rolling(window=window, center=True).mean()
    plt.plot(full_df['date'], rolling_mean, color=colors[i], label=labels[i], linewidth=2)

plt.title('Støjvurdering med Glidende Gennemsnit', fontsize=16)
plt.xlabel('Dato')
plt.ylabel('Temperatur (°C)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('glidende_gennemsnit.png')
plt.show()

# Del 4: Spektralanalyse for at vurdere støjens frekvensindhold
# Brug kun meantemp data for spektralanalysen
signal_data = full_df['meantemp'].dropna().values

# Beregn spektraltætheden (Power Spectral Density)
f, Pxx = signal.periodogram(signal_data, fs=1)  # antager 1 sample per dag

plt.figure(figsize=(12, 6))
plt.semilogy(f, Pxx)
plt.title('Spektraltæthed af Temperaturdata', fontsize=16)
plt.xlabel('Frekvens [cykler/dag]')
plt.ylabel('Power Spektraltæthed')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spektraltaethed.png')
plt.show()

# Del 5: Beregning af Signal-to-Noise Ratio (SNR) med forskellige filterstørrelser
plt.figure(figsize=(14, 10))

# Forskellige vinduesstørrelser for glidende gennemsnit
window_sizes = [3, 7, 14, 30]
snr_values = []

# Plot støjestimering for forskellige vinduesstørrelser
for i, window in enumerate(window_sizes):
    # Beregn glidende gennemsnit (signal)
    signal_estimate = full_df['meantemp'].rolling(window=window, center=True).mean()
    
    # Beregn støj ved at fratrække signal fra originale data
    noise_estimate = full_df['meantemp'] - signal_estimate
    
    # Beregn SNR (signal-to-noise ratio) i dB
    signal_power = np.nanmean(signal_estimate**2)
    noise_power = np.nanmean(noise_estimate**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    snr_values.append(snr)
    
    # Plot støjestimering
    plt.subplot(len(window_sizes), 1, i+1)
    plt.plot(full_df['date'], noise_estimate, 'r-', alpha=0.7)
    plt.title(f'Støjestimering (Vindue: {window} dage, SNR: {snr:.2f} dB)')
    plt.ylabel('Temperatur (°C)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('støjestimering.png')
plt.show()

# Del 6: Visualisér fordelingen af støj for det bedste filter
best_window = window_sizes[np.argmax(snr_values)]
signal_estimate = full_df['meantemp'].rolling(window=best_window, center=True).mean()
noise_estimate = full_df['meantemp'] - signal_estimate

plt.figure(figsize=(12, 6))
sns.histplot(noise_estimate.dropna(), kde=True)
plt.title(f'Fordeling af Støj (Vindue: {best_window} dage)', fontsize=16)
plt.xlabel('Støj (°C)')
plt.ylabel('Frekvens')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('støjfordeling.png')
plt.show()

# Del 7: Visualisér luftfugtighed (humidity) for sammenligning af støjmønstre
plt.figure(figsize=(14, 8))
plt.plot(full_df['date'], full_df['humidity'], label='Luftfugtighed')
plt.title('Daglig Luftfugtighed i Delhi (2013-2017)', fontsize=16)
plt.xlabel('Dato')
plt.ylabel('Luftfugtighed (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('luftfugtighed_serie.png')
plt.show()

# Del 8: Sammenlign støj i temperatur og luftfugtighed
humidity_signal = full_df['humidity'].rolling(window=best_window, center=True).mean()
humidity_noise = full_df['humidity'] - humidity_signal

plt.figure(figsize=(14, 10))

# Støj i temperatur
plt.subplot(2, 1, 1)
plt.plot(full_df['date'], noise_estimate, 'r-', alpha=0.7)
plt.title(f'Støj i Temperatur (Vindue: {best_window} dage)')
plt.ylabel('Temperatur (°C)')
plt.grid(True, alpha=0.3)

# Støj i luftfugtighed
plt.subplot(2, 1, 2)
plt.plot(full_df['date'], humidity_noise, 'b-', alpha=0.7)
plt.title(f'Støj i Luftfugtighed (Vindue: {best_window} dage)')
plt.ylabel('Luftfugtighed (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sammenligning_støj.png')
plt.show()

# Del 9: Korrelation mellem støj i forskellige variable
noise_df = pd.DataFrame({
    'date': full_df['date'],
    'temp_noise': noise_estimate,
    'humidity_noise': humidity_noise
}).dropna()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='temp_noise', y='humidity_noise', data=noise_df, alpha=0.5)
plt.title('Korrelation mellem Støj i Temperatur og Luftfugtighed', fontsize=16)
plt.xlabel('Støj i Temperatur (°C)')
plt.ylabel('Støj i Luftfugtighed (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('støj_korrelation.png')
plt.show()

# Beregn og vis korrelationskoefficient
correlation = noise_df['temp_noise'].corr(noise_df['humidity_noise'])
print(f"Korrelation mellem støj i temperatur og luftfugtighed: {correlation:.4f}")

# Del 10: Sammenfatning af støjniveau for alle variable
variables = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
window = best_window  # Brug det bedste vindue fra tidligere analyse

plt.figure(figsize=(14, 10))
snr_all = {}

for i, var in enumerate(variables):
    # Beregn signal og støj
    signal_est = full_df[var].rolling(window=window, center=True).mean()
    noise_est = full_df[var] - signal_est
    
    # Beregn SNR
    signal_power = np.nanmean(signal_est**2)
    noise_power = np.nanmean(noise_est**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    snr_all[var] = snr
    
    # Plot støj for hver variabel
    plt.subplot(len(variables), 1, i+1)
    plt.plot(full_df['date'], noise_est, alpha=0.7)
    plt.title(f'Støj i {var} (SNR: {snr:.2f} dB)')
    plt.ylabel(var)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('støj_alle_variable.png')
plt.show()

# Vis SNR-værdier for alle variable
for var, snr in snr_all.items():
    print(f"SNR for {var}: {snr:.2f} dB")
