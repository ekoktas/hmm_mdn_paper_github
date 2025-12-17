import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp

def analyze_distribution(file_path, feature, n_components=3):
    df = pd.read_csv(file_path)
    data = df[feature].dropna().values.reshape(-1, 1)
    data_log = np.log1p(data)

    # Plot histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=100, alpha=0.7)
    plt.title(f'{feature} (raw)')
    plt.subplot(1, 2, 2)
    plt.hist(data_log, bins=100, alpha=0.7)
    plt.title(f'log(1 + {feature})')
    plt.show()

    # Fit GMMs
    gmm_raw = GaussianMixture(n_components=n_components, random_state=0).fit(data)
    gmm_log = GaussianMixture(n_components=n_components, random_state=0).fit(data_log)

    # Compute negative log-likelihood
    nll_raw = -gmm_raw.score(data) * len(data)
    nll_log = -gmm_log.score(data_log) * len(data_log)

    # KS statistic (fit sample vs. data)
    sample_raw = gmm_raw.sample(len(data))[0]
    sample_log = gmm_log.sample(len(data_log))[0]
    ks_raw = ks_2samp(data.flatten(), sample_raw.flatten()).statistic
    ks_log = ks_2samp(data_log.flatten(), sample_log.flatten()).statistic

    print(f"{file_path} - {feature}")
    print(f"  Raw: NLL={nll_raw:.2f}, KS={ks_raw:.3f}")
    print(f"  Log: NLL={nll_log:.2f}, KS={ks_log:.3f}")

# Analyze both datasets and both features
for file in ['real_data/df_raw_HTTP.csv', 'real_data/df_raw_UDP_GOOGLE_HOME.csv']:
    for feature in ['payload_length', 'time_diff']:
        analyze_distribution(file, feature)