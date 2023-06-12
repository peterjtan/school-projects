import tarfile
import os
import subprocess
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast

from laser import CustomLaser

PADDING_SIZE = 512

class SentencePairDataset(Dataset):
    def __init__(self, lang_pairs, dataset_type, device=None):
        super().__init__()

        assert dataset_type in ['dev', 'train', 'test']

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.df, self.ind_langpair_map = self.get_dataframe(lang_pairs, dataset_type)
        self.target_sentences = self.df['translation'].to_numpy()
        self.source_sentences = self.df['original'].to_numpy()
        self.z_mean_scores = self.df['z_mean'].to_numpy()


    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, ind):
        target_sentence = self.target_sentences[ind]
        source_sentence = self.source_sentences[ind]
        
        for k in sorted(self.ind_langpair_map):
            langcode = self.ind_langpair_map[k]
            if ind < k:
                break

        # Get tokenized input by calling implementation in subclass
        result = self.get_item_impl(target_sentence, source_sentence, langcode)

        z_mean_score = torch.as_tensor(self.z_mean_scores[ind],
                                       device=self.device,
                                       dtype=torch.float)

        return result, z_mean_score


    def get_item_impl(self, target_sentence, source_sentence, langcode):
        raise NotImplementedError('This is a virtual function. Subclass should implement this!')


    def get_dataframe(self, lang_pairs, dataset_type):
        df_list = []
        ind_langpair_map = {}

        total_ind = 0
        for lang_pair in lang_pairs:
            lang_pair_4char = f'{lang_pair[0:2]}{lang_pair[3:5]}'

            if dataset_type == 'test':
                tar_filename = f'data/{lang_pair}_test.tar.gz'
                data_filename = f'{dataset_type}20.{lang_pair_4char}.df.short.tsv'
            else:
                tar_filename = f'data/{lang_pair}.tar.gz'
                data_filename = f'{dataset_type}.{lang_pair_4char}.df.short.tsv'

            with tarfile.open(tar_filename, 'r') as tar:
                for member in tar.getmembers():
                    if data_filename in member.name:
                        print(f'Processing {member.name} ...')
                        f = tar.extractfile(member)
                        df = pd.read_csv(f, sep='\\t', on_bad_lines='warn', engine='python')
                        total_ind += df.shape[0]
                        df_list.append(df)

            ind_langpair_map[total_ind] = lang_pair[0:2]

        df = pd.concat(df_list)
        df.reset_index(inplace=True)

        return df, ind_langpair_map


class SentencePairDatasetForBERT(SentencePairDataset):
    def __init__(self, lang_pairs, dataset_type, device=None):
        super().__init__(lang_pairs, dataset_type, device)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    def get_item_impl(self, target_sentence, source_sentence, langcode):
        # Tokenize input using BertTokenizer
        result = self.tokenizer(target_sentence,
                                source_sentence,
                                padding='max_length',
                                return_tensors='pt')

        # Transfer to correct device
        for k, v in result.items():
            result[k] = v.to(self.device)

        return result


class SentencePairDatasetForLaser(SentencePairDataset):
    def __init__(self, lang_pairs, dataset_type, device=None):
        super().__init__(lang_pairs, dataset_type, device)

        self.laser_model_path = os.path.join(os.getcwd(), 'laser_model')
        self.download_models()

        self.laser = CustomLaser()

    def get_item_impl(self, target_sentence, source_sentence, langcode):
        with torch.no_grad():
            result = self.laser.embed_sentences([target_sentence, source_sentence],
                                                lang=['en', langcode])

            encoder_out = result[0]['encoder_out'][0]

            if result[0]['encoder_padding_mask'] is not None:
                output_lengths = torch.sum(~result[0]['encoder_padding_mask'], dim=0)
                output_lengths = list(output_lengths.detach().cpu().numpy())
            else:
                output_lengths = [encoder_out.shape[0]] * 2

            valid_tensors = []
            for i, output_length in enumerate(output_lengths):
                valid_tensors.append(encoder_out[0:output_length, i, :])

            result = torch.vstack(valid_tensors).to(self.device)
            result = F.pad(result, pad=(0, 0, 0, PADDING_SIZE - result.shape[0]))

            length = sum(output_lengths)

        return result, length

    def download_models(self):
        if not os.path.exists(self.laser_model_path):
            os.mkdir(self.laser_model_path)
        # If laser_model folder presents, assume that the model has been downloaded
        else:
            return

        print(f'Running python -m laserembeddings download-models {self.laser_model_path} ...')
        output = subprocess.run(f'python -m laserembeddings download-models "{self.laser_model_path}"',
                                capture_output=True,
                                check=True)

        print(output.stdout)


class SentencePairDatasetForLaBSE(SentencePairDataset):
    def __init__(self, lang_pairs, dataset_type, device=None):
        super().__init__(lang_pairs, dataset_type, device)

        self.labse_model_path = os.path.join(os.getcwd(), 'labse_model')
        if not os.path.exists(self.labse_model_path):
            print('\nError: Missing pre-trained model')
            print('Please download the zip file from https://drive.google.com/file/d/1gKX_GS2ypZ4KOvtnh9WN15zxO89T5CWo/view?usp=sharing')
            print('Then unzip into \'./labse_model\' folder')
            print('It may also be necessary to add \'"model_type": "bert"\' to \'config.json\' within \'./labse_model\'')
            exit()

        self.max_seq_length = PADDING_SIZE

        self.tokenizer = BertTokenizerFast.from_pretrained(self.labse_model_path,
                                                       do_lower_case=False)


    def get_item_impl(self, target_sentence, source_sentence, lang_code):
        # Tokenize input using pre-trained tokenizer
        token_results = self.tokenizer(target_sentence, 
                                       source_sentence, 
                                       add_special_tokens=True,
                                       padding='max_length', 
                                       max_length=self.max_seq_length)
        # Transfer to the correct device
        for k, v in token_results.items():
            token_results[k] = torch.tensor(v).to(self.device)

        return token_results


# def get_sentence_pairs(df):
#     """
#     Returns a numpy array with sentences pairs: 
#     [
#         [target, source],
#         [target, source],
#         ...
#     ]
#     """
#     return df[['translation', 'original']].to_numpy()


def train(dataloader, model, criterion, optimizer,
          output_model_name, num_epochs=10):

    print('-----')
    print('Begin training ...')

    num_batches = len(dataloader)
    print(f'# of batches: {num_batches}')

    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = datetime.now()

        for model_input, z_mean_scores in tqdm(dataloader):
            optimizer.zero_grad()

            output = model(model_input).squeeze()
            loss = criterion(output, z_mean_scores)
            epoch_loss += loss

            loss.backward()
            optimizer.step()

        print(f'Epoch loss (MSE): {epoch_loss / num_batches}')

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        print(f'Time for epoch #{epoch}: {total_time}')

        # Checkpoint: save model state dict
        path = os.path.join(os.getcwd(), 'model')
        if not os.path.exists(path):
            os.mkdir(path)
        
        path = os.path.join(path, output_model_name)
        torch.save(model.state_dict(), path)


def test(dataloader, model):
    print('-----')
    print('Begin testing ...')

    model.eval()

    num_batches = len(dataloader)
    print(f'# of batches: {num_batches}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mseloss_fn = torch.nn.MSELoss().to(device=device)

    start_time = datetime.now()
    output_list = []
    z_mean_score_list = []

    for model_input, z_mean_scores in tqdm(dataloader):
        output = model(model_input).squeeze()
        output_list.append(output)
        z_mean_score_list.append(z_mean_scores)

    output_tensor = torch.cat(output_list, dim=0)
    z_mean_score_tensor = torch.cat(z_mean_score_list, dim=0)

    # RMSE
    rmse = torch.sqrt(mseloss_fn(output_tensor, z_mean_score_tensor))

    # Pearson Correlation
    vx = output_tensor - torch.mean(output_tensor)
    vy = z_mean_score_tensor - torch.mean(z_mean_score_tensor)
    peason_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    print(f'RMSE: {rmse}; Peason Correlation: {peason_correlation}')

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f'Total time: {total_time}')
