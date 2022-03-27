import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader
from gtzan_dataset import GTZANDataset
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "Data/features_30_sec_final.csv"
AUDIO_DIR = "Data/genres_original"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 # -> 1 second of audio


def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), torch.tensor(targets).to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets).to(device)

        # backpropagate loss and update weights
        optimiser.zero_grad()   # reset gradient to zero at each iteration
        loss.backward()
        optimiser.step()    # update weights

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("--------------")
    print(f"Completed training for {i+1} epochs")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # instantiate dataset object and create data loader
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
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    train_data_loader = create_data_loader(train_data=gtzan, batch_size=BATCH_SIZE)

    # build model
    cnn = CNNNetwork().to(device)

    # instantiate loss funcion and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr = LEARNING_RATE)

    # train model
    train(
        cnn,
        train_data_loader,
        loss_fn,
        optimiser,
        device,
        EPOCHS
    )

    # save the model
    torch.save(cnn.state_dict(), "cnn.pth")
    print("Trained model is stored at cnn.pth")