import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config
from .model import Transformer
from .tokenizer.bpe.bpe import BytepairEncoding


class TinyShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer():
    def __init__(self, config):
        self.config = config
        self.bpe = BytepairEncoding()

    def preprocess(self):
        # train the tokenizer first
        self.bpe.train(self.config.data_path, self.config.vocab_size)

    def create_dataset(self):
        """
            1. Load the text file
            2. tokenize the data
            3. Create batches
            4. Add EOS token on both input and output and add eos in tokenizer too
        """
        text = open(self.config.data_path, "r").read()
        print("Tokenizing the data...")
        tokens = self.bpe.encode(text)
        # create input and output
        inputs = []
        targets = []
        for i in range(0, len(tokens), self.config.seq_length-1):
            # seq length -1 since we will also add <eos> token
            inp = tokens[i:i+self.config.seq_length-1]
            inp.append(self.config.eos_token_id)
            out = tokens[i+1:i+self.config.seq_length]
            out.append(self.config.eos_token_id)
            if len(inp) != self.config.seq_length or len(out) != self.config.seq_length:
                # if the last batch doesn't match seq length, we just ignore. But we can pad and use that 
                continue
            inputs.append(torch.Tensor(inp))
            targets.append(torch.Tensor(out))

        # zip input andoutput
        dataset = list(zip(inputs, targets))

        return dataset

    def create_dataloader(self):
        if os.path.exists("models/dataloader.pth"):
            return torch.load("models/dataloader.pth")
        data = self.create_dataset()
        dataset = TinyShakespeareDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        torch.save(dataloader, "models/dataloader.pth")
        return dataloader

    def create_model(self):
        return Transformer(self.config)

    def train(self):
        # create model
        print("creatiuung model")
        model = self.create_model()
        model = model.to(self.config.device)

        # define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, betas=[self.config.beta_1, self.config.beta_2], eps=self.config.epsilon,)

        # create dataloader
        print("creating dataloader")
        dataloader = self.create_dataloader()

        for epoch in range(self.config.num_epochs):
            # Use tqdm to create a progress bar
            with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as progress_bar:
                for i, (inputs, outputs) in progress_bar:
                    # get outputs
                    inputs = inputs.to(self.config.device)
                    outputs = outputs.to(self.config.device)
                    inputs = inputs.long()
                    outputs = outputs.long()
                    targets = model(inputs, outputs)

                    # loss
                    # need to flatten the tokensinput
                    # Output: [N, C] → [total_tokens, vocab_size]
                    # target: [N]   → [total_tokens]

                    targets = targets.view(-1, targets.size(-1))
                    outputs = outputs.view(-1)
                    loss = criterion(targets, outputs)
                    # backward prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update progress bar with the current loss
                    progress_bar.set_postfix(loss=loss.item())

                    # save model and config at the end of each epoch
                    if i == len(dataloader) - 1:
                        torch.save(model.state_dict(), self.config.model_path)
                        # Assuming save_config is a method of Config class to save the config
                        # self.config.save_config(self.config.config_path)


trainer = Trainer(Config())
