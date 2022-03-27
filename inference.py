import torch
import torchaudio
from cnn import CNNNetwork
from gtzan_dataset import GTZANDataset
from cnn_train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]

def predict(model, X, y, class_mapping):
    model.eval()    # train <-> eval: changes how model behave (e.g. no dropout, ...)
    with torch.no_grad():
        predictions = model(X)
        # tensor (1, 10) -> [ [0.1, 0.04, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[y]
    
    return predicted, expected



if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    # load gtzan validation dataset
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=20,
        log_mels=True
    )

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    gtzan = GTZANDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mfcc,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device="cpu"
    )


    index = 0
    # get a sample from the gtzan dataset for inference
    X, y = gtzan[index][0], gtzan[index][1] # [batch_size, num_channels, freq, time]
    X.unsqueeze_(0) # insert an extra dimension at index 0
    print(X.shape)
    print(y)

    # make an inference
    predicted, expected = predict(cnn, X, y, class_mapping)
    
    print(f"Predicted: {predicted}")
    print(f"Expected: {expected}")