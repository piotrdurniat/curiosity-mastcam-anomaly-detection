from argparse import ArgumentParser
from utilis import train_model

def main():

    parser = ArgumentParser()
    parser.add_argument("--model", choices=["GAN", "VAE", "FLOW"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()
    print(args)

    train_model(model_name=args.model, epoch_number=args.epochs, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()