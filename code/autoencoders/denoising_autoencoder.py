"""
Pytorch implementation of the Autoencoder
For denoising the images
"""
import sys
import torch

sys.path.append("../")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import matplotlib.pyplot as plt
from torch import nn
from dataloader import load_mnist
from basic_autoencoder import Encoder, Decoder, save_checkpoint


def add_noise(inputs, noise_factor=0.3):
     noisy = inputs + torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy, 0., 1.)

     return noisy


if __name__ == "__main__":

    print(f"Using {device} as the accelerator")
    try:
        # try loading checkpoint
        checkpoint = torch.load('../models/denoising_autoencoder.pth.tar')
        print("Found Checkpoint :)")
        encoder = checkpoint["encoder"]
        decoder = checkpoint["decoder"]
        encoder.to(device)
        decoder.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")

        encoder = Encoder()
        decoder = Decoder()
        encoder.to(device)
        decoder.to(device)
        criterion = nn.MSELoss().to(device)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
        
        train_dataloader, _ = load_mnist()

        for epoch in range(10):
            for i, (image, _) in enumerate(train_dataloader):
                image.to(device)
                noisy_image = add_noise(image)
                noisy_image.to(device)
                encoded_image = encoder(noisy_image)
                decoded_image = decoder(encoded_image)
                
                loss = criterion(decoded_image, image)
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()

                decoder_optimizer.step()
                encoder_optimizer.step()

                if i % 100 == 0 and i != 0:
                    print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")

            save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path='../models/denoising_autoencoder.pth.tar')
    
    # do reconstruction
    _, test_dataloader = load_mnist(batch_size=1)
    decoded_image    = None

    for i, (image, _) in enumerate(test_dataloader):
        image.to(device)
        noisy_image = add_noise(image)
        noisy_image.to(device)
        encoded_image = encoder(noisy_image)
        decoded_image = decoder(encoded_image)
        break
    
    image = image.reshape(28, 28).detach().numpy()
    noisy_image = noisy_image.reshape(28, 28).detach().numpy()
    reconstructed_image = decoded_image.reshape(28, 28).detach().numpy()

    plt.gray()
    fig, axis = plt.subplots(1, 3)
    axis[0].imshow(image)
    axis[1].imshow(noisy_image)
    axis[2].imshow(reconstructed_image)
    plt.savefig(f"./img/denoising_autoencoder.png", dpi=600)
    plt.show()
