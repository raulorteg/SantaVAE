import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.modules.loss import BCELoss

sys.path.append(".")

from GenerativeSanta.plots import image_grid
from GenerativeSanta.utils import SantaDataset

# from GenerativeSanta.models import SantaFinder


def main(n_epochs, lr, batch_size, resize, val_freq):
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

    # with open("results/results.txt", "w") as f:
    #     print("epoch,train_loss,train_acc,test_loss,test_acc", file=f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\t -> Using device: {device}")

    # net = SantaFinder(input_shape=input_shape).to(device)
    # optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    # criterion = torch.nn.BCEWithLogitsLoss()

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for epoch in range(1, n_epochs + 1):
        # training loop
        running_loss = 0.0
        acc_buffer = []
        # net.train()
        for sample in dataloader:
            batch_size = sample["image"].shape[1]

            images, labels = sample["image"].to(device), sample["label"].to(device)

            img = images[0]
            print(img.shape)
            numpy_image = (images[0].cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                "uint8"
            )
            pil_image = Image.fromarray(numpy_image.astype("uint8"))
            pil_image.save("a.png")
            exit()

            # optimizer.zero_grad()

            # predictions = net(images)
            # preds = predictions["preds"]

            # loss = criterion(preds.double(), labels.unsqueeze(1).double())

            # loss.backward()
            # optimizer.step()

            # running_loss += loss.cpu().detach().item()/batch_size
            # acc_buffer.append(compute_accuracy(preds.squeeze().detach().cpu().numpy(), labels.squeeze().detach().cpu().numpy()))

        # train_loss.append(running_loss/len(dataset))
        # train_acc.append(np.array(acc_buffer).mean())

    print("Done!")


if __name__ == "__main__":
    import argparse

    # parsing user input
    # example: python main.py --n_epochs=100 --lr=0.0001 --batch_size=32 --latent_size=100 --resize=128 --val_freq=1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", help="Number of epochs (defaults 100)", default=10, type=int
    )
    parser.add_argument(
        "--lr", help="Learning rate (defaults 0.002)", default=0.002, type=float
    )
    parser.add_argument(
        "--batch_size", help="Batch size (defaults 32)", default=32, type=int
    )
    parser.add_argument(
        "--resize",
        help="Pixels of the resized image (defaults to 128)",
        default=128,
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
    )
