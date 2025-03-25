import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Indlæs datasættene (sørg for, at filerne ligger i samme mappe som scriptet)
train_df = pd.read_csv("DailyDelhiClimateTrain.csv")
test_df = pd.read_csv("DailyDelhiClimateTest.csv")

# Konverter dato-kolonnen til datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Funktion til at anvende et lowpass filter (glidende gennemsnit)
def apply_lowpass_filter(df, column, window_size):
    df[f'{column}_filtered'] = df[column].rolling(window=window_size, min_periods=1).mean()

# Funktion til at anvende Fourier-transformation
def apply_fourier_transform(df, column):
    df[f'{column}_fourier'] = np.fft.fft(df[column].fillna(0))

# Funktion til linjediagrammer for meantemp
def plot_meantemp(df, title):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=14)
    sns.lineplot(x=df['date'], y=df['meantemp'], label='Mean Temperature')
    sns.lineplot(x=df['date'], y=df['meantemp_filtered'], label='Filtered Mean Temperature')
    plt.xlabel('Date')
    plt.ylabel('Mean Temperature')
    plt.legend()
    plt.show()

# Funktion til at plotte Fourier-transformation
def plot_fourier(df, column, title):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=14)
    freqs = np.fft.fftfreq(len(df[column].dropna()))
    fourier = np.abs(df[f'{column}_fourier'])
    sns.lineplot(x=freqs, y=fourier, label='Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Anvend lowpass filter og Fourier-transformation på trænings- og testdatasæt
apply_lowpass_filter(train_df, 'meantemp', window_size=10)
apply_lowpass_filter(test_df, 'meantemp', window_size=10)
apply_fourier_transform(train_df, 'meantemp')
apply_fourier_transform(test_df, 'meantemp')

# Plot for træningsdata
plot_meantemp(train_df, "Linjediagram for Mean Temperature (Træningsdatasæt)")
plot_fourier(train_df, 'meantemp', "Fourier Transform for Mean Temperature (Træningsdatasæt)")

# Plot for testdata
plot_meantemp(test_df, "Linjediagram for Mean Temperature (Testdatasæt)")
plot_fourier(test_df, 'meantemp', "Fourier Transform for Mean Temperature (Testdatasæt)")