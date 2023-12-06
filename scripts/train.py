import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.modules.loss import BCELoss

sys.path.append(".")

from GenerativeSanta.plots import image_grid
from GenerativeSanta.utils import SantaDataset
from GenerativeSanta.models import CVAE


def main(n_epochs, lr, batch_size, resize, val_freq, beta):
    # load datasets
    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ]
    )

    transform_eval = transforms.Compose(
        [
            transforms.Resize((resize, resize), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )

    dataset = SantaDataset(txt_file="data/training.txt", transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    dataset_eval = SantaDataset(txt_file="data/testing.txt", transform=transform_eval)
    dataloader_eval = torch.utils.data.DataLoader(dataset, batch_size=32)

    for sample in dataloader:
        input_shape = sample["image"].shape
        break

    with open("results/results.txt", "w") as f:
        print("epoch,train_loss,train_acc,test_loss,test_acc", file=f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\t -> Using device: {device}")

    net = CVAE(input_size=input_shape, latent_size=20).to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for epoch in range(1, n_epochs + 1):
        # training loop
        train_loss = 0.0
        acc_buffer = []
        # net.train()
        for sample in dataloader:
            batch_size = sample["image"].shape[1]

            images, labels = sample["image"].to(device), sample["label"].to(device)

            # img = images[0]
            # print(img.shape)
            # numpy_image = (images[0].cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            #     "uint8"
            # )
            # pil_image = Image.fromarray(numpy_image.astype("uint8"))
            # pil_image.save("a.png")
            # exit()

            optimizer.zero_grad()

            outdict = net(x=images, c=labels)
            pz, qz, z, x, xhat = outdict["pz"], outdict["qz"], outdict["z"], outdict["x"], outdict["xhat"]

            # compute loss
            MSE = torch.mean((xhat - x)**2)
            KLD = -beta * torch.mean(torch.mean(1 + torch.log(qz.sigma**2) - qz.mu**2 - torch.exp(qz.sigma**2)))
            loss = MSE + KLD

            
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()

        scheduler.step()
        
        # Loop to generate images (done every EVAL_FREQ)
        if (epoch % val_freq == 0) or (epoch == 1):
            net.eval()
            for sample in dataloader_eval:
                optimizer.zero_grad()
                images = sample["image"].to(device)
                encodings = sample["label"].to(device)
                outdict = net(images, encodings)
                x, xhat = outdict["x"], outdict["xhat"]
                break
            
            images = []
            for img in xhat:
                img = (255*img).cpu().detach().numpy().astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))
                pil_image = Image.fromarray(img).convert("RGB")
                images.append(pil_image)
            
            grid = image_grid(images, 4, 8, resize, resize)
            grid.save(f"results/iter_{epoch}.png")


        # update lr scheduler
        last_lr = scheduler.get_last_lr()[0]

        # print some info to the terminal and to the traning results file
        print(f"epoch:{epoch}, loss:{train_loss}, mse: {MSE}, kld: {KLD}, lr: {last_lr}")
        with open('results/training.txt', 'a') as f:
            print(epoch,train_loss,MSE.item(),KLD.item(),last_lr, file=f)

    print("Done!")


if __name__ == "__main__":
    import argparse

    # parsing user input
    # example: python main.py --n_epochs=100 --lr=0.0001 --batch_size=32 --latent_size=100 --resize=128 --val_freq=1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", help="Number of epochs (defaults 100)", default=100, type=int
    )
    parser.add_argument(
        "--lr", help="Learning rate (defaults 0.002)", default=0.002, type=float
    )
    parser.add_argument(
        "--beta", help="Beta for weighting the KL term.", default=1.0, type=float
    )
    parser.add_argument(
        "--batch_size", help="Batch size (defaults 32)", default=32, type=int
    )
    parser.add_argument(
        "--resize",
        help="Pixels of the resized image (defaults to 128)",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--val_freq",
        help="Validation frequency (defaults to every 5 epochs)",
        default=1,
        type=int,
    )
    args = parser.parse_args()

    main(
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        resize=args.resize,
        val_freq=args.val_freq,
        beta=args.beta,
    )
