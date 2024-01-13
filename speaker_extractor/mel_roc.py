import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import librosa.display
import librosa
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_and_compute_melspectrogram(file_path, n_mels=128, n_fft=48, hop_length=512):
    # Load the WAV file
    y, sr = librosa.load(file_path)  # Load with the original sampling rate

    # Compute Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Normalize S_dB to 0-1 range
    S_dB_min = S_dB.min()
    S_dB_max = S_dB.max()
    S_dB_normalized = (S_dB - S_dB_min) / (S_dB_max - S_dB_min)

    return S_dB_normalized


def wav_to_binary_array(file_path, threshold = 0.02):
    y = load_and_compute_melspectrogram(file_path)
    # y, sr = librosa.load(file_path)
    binary_array = np.where(y > threshold, 1, 0)

    return binary_array




def calculate_roc_auc(energy, ground_truth):
    # Flatten both arrays for comparison
    energy_flat = energy.flatten()
    ground_truth_flat = ground_truth.flatten()
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(ground_truth_flat, energy_flat)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# load GT
file_path = '/Users/netaglazer/PycharmProjects/Create-dataset/speaker_extractor/Dan_Meridor_target-SD_vad.wav'
binary_array = wav_to_binary_array(file_path)  # Visualize the waveform
print()

# load FRONT / (Any other prediction)
file_path = '/Users/netaglazer/PycharmProjects/Create-dataset/Dataset_V3_Meridor/1/front.wav'
y, sr = librosa.load(file_path)
mel = load_and_compute_melspectrogram(file_path)


fpr, tpr, roc_auc = calculate_roc_auc(mel, binary_array)
print(f"ROC AUC: {roc_auc}")

# Visualize the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Example usage











