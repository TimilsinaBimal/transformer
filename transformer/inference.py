import torch

from .config import Config
from .model import Transformer
from .tokenizer.bpe.bpe import BytepairEncoding


class Inference():
    def __init__(self):
        self.config = Config()
        model_path = "models/model.pth"
        self.model = Transformer(self.config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.config.device)
        self.model.eval()
        self.bpe = BytepairEncoding()

    def generate(self, prompt, max_length=50):
        # Tokenize the prompt
        tokens = self.bpe.encode(prompt)
        # Initially, outputs start with the last token of input
        outputs = torch.tensor([tokens[-1]], device=self.config.device).unsqueeze(0).long()
        tokens = torch.tensor(tokens, device=self.config.device).unsqueeze(0).long()

        # Pad input to match model's expected input size
        pad_length = self.config.seq_length - tokens.shape[1]
        if pad_length > 0:
            padding = torch.full((1, pad_length), self.config.eos_token_id, device=self.config.device)
            tokens = torch.cat((tokens, padding), dim=1)
        else:
            tokens = tokens[:, :self.config.seq_length]  # Truncate if longer than seq_length

        for _ in range(max_length):
            # Get model prediction
            with torch.no_grad():
                tokens_to_pass = tokens[:, :outputs.shape[1]]
                tokens_to_pass = tokens_to_pass.to(self.config.device)
                outputs = outputs.to(self.config.device)
                # Pass the tokens and outputs to the model
                output = self.model(tokens_to_pass, outputs)

                # Get the most probable next token
                last_token = output[:, -1, :]
                next_token = torch.argmax(last_token, dim=-1)

                # decode and print
                print(self.bpe.decode(next_token.tolist()), end=" ")
                # Stop if EOS token is generated
                if next_token.item() == self.config.eos_token_id:
                    break

                # Append the new token to the outputs
                outputs = torch.cat((outputs, next_token.unsqueeze(0)), dim=1)

        # Decode the output
        output_sequence = outputs.squeeze(0).tolist()
        output_text = self.bpe.decode(output_sequence)
        print(output_text)
        return output_text
