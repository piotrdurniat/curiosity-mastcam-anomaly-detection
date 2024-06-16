from argparse import ArgumentParser


def main():

    parser = ArgumentParser()
    parser.add_argument("--model", choices=["GAN", "VAE", "FLOW"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()
    print(args)

if __name__ == "__main__":
    main()