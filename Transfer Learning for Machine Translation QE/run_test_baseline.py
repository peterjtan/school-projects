import os

import torch
from torch.utils.data import DataLoader

from util import SentencePairDatasetForBERT, test
from baseline import BertPredictorEstimator

BATCH_SIZE = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f'Using device {device} ...')


def main():
    # Prepare testing data
    dataset = SentencePairDatasetForBERT(lang_pairs=['en-de', 'en-zh'],
                                         dataset_type='test',
                                         device=device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Model and other initialization
    state_dict = torch.load(f=os.path.join('model', 'PredictorEstimator_bert.pt'),
                            map_location=device)

    model = BertPredictorEstimator().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    test(dataloader, model)


if __name__ == '__main__':
    main()
