import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import librosa.display



def calculate_normalized_pointwise_energy(file_path):
    # Load the WAV file
    y, sr = librosa.load(file_path)

    # Calculate the pointwise energy
    energy = np.square(y)

    # Normalize the energy to the range [0, 1]
    max_energy = np.max(energy)
    if max_energy > 0:
        normalized_energy = energy / max_energy
    else:
        normalized_energy = energy  # to avoid division by zero if max_energy is 0

    return normalized_energy




def wav_to_binary_array(file_path, threshold = 0.02):
    y, sr = librosa.load(file_path)
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


# load FRONT / (Any other prediction)
file_path = '/Users/netaglazer/PycharmProjects/Create-dataset/Dataset_V3_Meridor/1/front.wav'
normalized_energy = calculate_normalized_pointwise_energy(file_path)



fpr, tpr, roc_auc = calculate_roc_auc(normalized_energy, binary_array)
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

