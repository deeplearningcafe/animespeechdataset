import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='choose api to use'
    )
    parser.add_argument('--api', default="espnet", type=str, required=False, choices=["espnet", "reazonspeech"],
        help='api type')
    args = parser.parse_args()
    print(args.api)
    if args.api == "espnet":
        try:
            from .embeddings_api import espnet_api
        except:
            raise ImportError('The espnet library is not installed') 

        espnet_api()
    elif args.api == "reazonspeech":
        try:
            from .asr_api import reazonspeech_api
        except ImportError:
            raise ImportError('The reazonspeech library is not installed') 
        reazonspeech_api()
        