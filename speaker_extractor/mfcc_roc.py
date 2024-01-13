import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import librosa.display

def load_and_normalize_mfcc(file_path, n_mfcc=13):
    # Load the audio file
    y, sr = librosa.load(file_path)
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Normalize MFCCs to 0-1 range
    mfccs_min = np.min(mfccs, axis=1).reshape(-1, 1)
    mfccs_max = np.max(mfccs, axis=1).reshape(-1, 1)
    mfccs_normalized = (mfccs - mfccs_min) / (mfccs_max - mfccs_min)

    return mfccs_normalized, sr



def wav_to_binary_array(file_path, threshold = 0.02):
    # y, sr = librosa.load(file_path)
    y, sr = load_and_normalize_mfcc(file_path)
    binary_array = np.where(y > threshold, 1, 0)

    return binary_array



def visualize_mfcc(mfccs_normalized):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs_normalized, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Normalized MFCC')
    plt.tight_layout()
    plt.show()

def calculate_roc_auc(mfccs_normalized, ground_truth):
    # Flatten both arrays for comparison
    mfccs_flat = mfccs_normalized.flatten()
    ground_truth_flat = ground_truth.flatten()
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(ground_truth_flat, mfccs_flat)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# load GT
file_path = '/Users/netaglazer/PycharmProjects/Create-dataset/speaker_extractor/Dan_Meridor_target-SD_vad.wav'
binary_array = wav_to_binary_array(file_path)  # Visualize the waveform


# load FRONT / (Any other prediction)
file_path = '/Users/netaglazer/PycharmProjects/Create-dataset/Dataset_V3_Meridor/1/front.wav'
normalized_mfccs, _ = load_and_normalize_mfcc(file_path)
visualize_mfcc(normalized_mfccs)



fpr, tpr, roc_auc = calculate_roc_auc(normalized_mfccs, binary_array)
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

