from argparse import ArgumentParser

from utilistest import test_model


def main():

    parser = ArgumentParser()
    parser.add_argument("--model", choices=["GAN", "VAE", "FLOW"], required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    print(args)

    test_model(model_name=args.model, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
