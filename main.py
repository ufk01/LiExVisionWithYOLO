import argparse
from train import Train
from test import Test
from enums.config import GlobalValues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid FER Pipeline")
    parser.add_argument(
        "--mode", 
        type=str, 
        default=GlobalValues.TRAIN.value, 
        choices=[GlobalValues.TRAIN.value, GlobalValues.TEST.value, GlobalValues.CLEAN.value], 
        help="Operation mode: train, test, or clean"
    )
    args = parser.parse_args()

    if args.mode == "train":
        print("=== Training Process Initiated ===", flush=True)
        trainer = Train()
    elif args.mode == "test":
        print("=== Test Process Initiated ===", flush=True)
        tester = Test()