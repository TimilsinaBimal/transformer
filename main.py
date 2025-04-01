def train():
    from transformer.train import trainer
    trainer.train()


def inference():
    from transformer.inference import Inference
    inference = Inference()
    inference.generate("Before we proceed any further")


if __name__ == "__main__":
    inp = input("Do you want to train the model? (y for train/n for inference): ")
    if inp == "y":
        train()
    else:
        inference()
