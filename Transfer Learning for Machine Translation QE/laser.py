import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import util

from laserembeddings import Laser
from laserembeddings.encoder import SentenceEncoder
from laserembeddings.embedding import BPESentenceEmbedding

import os

BATCH_SIZE = 128
LASER_HIDDEN_SIZE = 1024
LSTM_HIDDEN_SIZE = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f'Using device {device} ...')


class CustomSentenceEncoder(SentenceEncoder):
    """
    Only overrides _process_batch to return all information from encoder.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override
    def encode_sentences(self, sentences):
        """
        Adapted from: https://github.com/yannvgn/laserembeddings/blob/ceb3818c998099d315a935210d3962640922fa8b/laserembeddings/encoder.py#L134
        """
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            results.append(self._process_batch(batch))
        return results

    # override
    def _process_batch(self, batch):
        """
        Adapted from: https://github.com/yannvgn/laserembeddings/blob/ceb3818c998099d315a935210d3962640922fa8b/laserembeddings/encoder.py#L83
        """
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()
        self.encoder.eval()
        return self.encoder(tokens, lengths)


class CustomBPESentenceEmbedding(BPESentenceEmbedding):
    """
    Only purpose is to replace encoder class with CustomSentenceEncoder.
    Adapted from: https://github.com/yannvgn/laserembeddings/blob/ceb3818c998099d315a935210d3962640922fa8b/laserembeddings/embedding.py#L24
    """
    def __init__(self,
                 encoder,
                 max_sentences=None,
                 max_tokens=12000,
                 stable=False,
                 cpu=False):

        self.encoder = CustomSentenceEncoder(
            encoder,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            sort_kind='mergesort' if stable else 'quicksort',
            cpu=cpu)


class CustomLaser(Laser):
    def __init__(self):
        laser_model_path = os.path.join(os.getcwd(), 'laser_model')
        encoder = os.path.join(laser_model_path, 'bilstm.93langs.2018-12-26.pt')

        super().__init__(bpe_codes=os.path.join(laser_model_path, '93langs.fcodes'),
                         bpe_vocab=os.path.join(laser_model_path, '93langs.fvocab'),
                         encoder=encoder)

        self.bpeSentenceEmbedding = CustomBPESentenceEmbedding(encoder=encoder, 
                                                               cpu=device.type == 'cpu')



class LaserPredictorEstimator(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=LASER_HIDDEN_SIZE,
                            hidden_size=LSTM_HIDDEN_SIZE,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(in_features=LSTM_HIDDEN_SIZE * 2,
                                out_features=1)


    def forward(self, laser_output_data):
        seq = laser_output_data[0]
        seq_lengths = laser_output_data[1]

        pack_seq = pack_padded_sequence(seq,
                                        seq_lengths,
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
    dataset = util.SentencePairDatasetForLaser(lang_pairs=['et-en', 'ro-en', 'ne-en'],
                                               dataset_type='dev',
                                               device=device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Model and other initialization
    model = LaserPredictorEstimator().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    util.train(dataloader, model, criterion, optimizer,
               output_model_name='PredictorEstimator_laser.pt',
               num_epochs=10)


if __name__ == '__main__':
    main()
