import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import List

import numpy as np

import util

from transformers import AutoModel

import os

BATCH_SIZE = 16
LaBSE_HIDDEN_SIZE = 768
LSTM_HIDDEN_SIZE = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f'Using device {device} ...')

class LaBSEPredictorEstimator(nn.Module):
    ##
    ## Pre-trained model from https://github.com/yang-zhang/labse-pytorch
    ##
    
    def __init__(self):
        super().__init__()

        print("Initializing LaBSE model...")
        labse_model_path = os.path.join(os.getcwd(), 'labse_model')
        self.labse_model = AutoModel.from_pretrained(labse_model_path).to(device)
        
        # Freeze parameters in LaBSE model
        for name, param in self.labse_model.named_parameters():
            param.requires_grad = False

        # Initialize additional layers
        self.lstm = nn.LSTM(input_size=LaBSE_HIDDEN_SIZE,
                            hidden_size=LSTM_HIDDEN_SIZE,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(in_features=LSTM_HIDDEN_SIZE * 2,
                                out_features=1)

        print('-----')
        print('Trainable parameters in LaBSEPredictorEstimator:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


    def forward(self, tokenizer_data):

        # Next 7 lines adapted from https://github.com/yang-zhang/labse-pytorch
        input_ids = tokenizer_data['input_ids']
        token_type_ids = tokenizer_data['token_type_ids']
        attention_mask = tokenizer_data['attention_mask']

        # input_ids = torch.tensor(input_ids).to(device)
        # token_type_ids = torch.tensor(token_type_ids).to(device)
        # attention_mask = torch.tensor(attention_mask).to(device)

        with torch.no_grad():
            result = self.labse_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        last_hidden_state = result['last_hidden_state']

        # Estimator: run through a Bi-LSTM
        lengths = torch.sum(attention_mask, dim=1).to("cpu")
        pack_seq = pack_padded_sequence(last_hidden_state,
                                        lengths,
                                        batch_first=True,
                                        enforce_sorted=False)

        output, _ = self.lstm(pack_seq)
        output, output_lens = pad_packed_sequence(output, batch_first=True)

        # (batch size, sequence length, 2 layers, hidden size)
        batch_size = min(len(output_lens), BATCH_SIZE)
        forward_last = output[range(batch_size), output_lens - 1, :LSTM_HIDDEN_SIZE]
        backward_front = output[:, 0, LSTM_HIDDEN_SIZE:]

        # (batch size, hidden size x 2)
        summary_vector = torch.cat((forward_last, backward_front), dim=1)

        output = self.linear(summary_vector)

        return output


def main():
    dataset = util.SentencePairDatasetForLaBSE(lang_pairs=['et-en', 'ro-en', 'ne-en'],
                                               dataset_type='dev',
                                               device=device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Model and other initialization
    model = LaBSEPredictorEstimator().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    util.train(dataloader, model, criterion, optimizer,
               output_model_name='PredictorEstimator_labse.pt',
               num_epochs=10)


if __name__ == '__main__':
    main()
