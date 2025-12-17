import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import io

def audio_to_spectrogram(
    file_path: str,
    sr: int | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(file_path, sr=sr)

    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to power spectrogram
    spectrogram = np.abs(stft) ** 2
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  # to decibel
        
    # normalize value to uint8
    spectrogram -= spectrogram.min()
    spectrogram /= spectrogram.max() + 1e-8
    spectrogram = (spectrogram * 255).astype(np.uint8)
    
    # Flip vertically so low frequencies are at the bottom
    spectrogram = np.flipud(spectrogram)

    # Plot spectrogram → image
    fig = plt.figure(figsize=(4, 4), dpi=100)
    plt.axis("off")
    plt.imshow(spectrogram, cmap='gray')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf).convert("RGB")
    