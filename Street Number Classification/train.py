import torch
import torch.nn as nn
import torch.optim as optim

import copy
from datetime import datetime
from tqdm.autonotebook import tqdm
import pickle
import os

from common import prepare_vgg_model, prepare_own_model
from dataset import prepare_dataset


NUM_EPOCH = 20

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(device)


def train(model, dataloader, dataset_size,
          optimizer, loss_fn, lr_scheduler, out_folder):

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    model.to(device)

    best_model_weights = None
    best_test_accuracy = 0.

    train_accuracy = []
    test_accuracy = []

    for epoch in range(NUM_EPOCH):
        print(f'Epoch {epoch + 1}/{NUM_EPOCH}')
        print(f'Started at {datetime.now()}')
        print('------------------------------')

        for phase in ['train', 'test']:
            is_train = phase == 'train'
            running_correct_preds = 0
            running_loss = 0.

            if is_train:
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = loss_fn(outputs, labels)

                    if is_train:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct_preds += torch.sum(preds == labels.data)

            # ReduceLROnPlateau scheduler needs validation loss
            if not is_train:
                lr_scheduler.step(running_loss)

            accuracy = running_correct_preds.double() / dataset_size[phase]
            print(f'Accuracy: {accuracy}')

            if phase == 'train':
                train_accuracy.append(accuracy)
            else:
                test_accuracy.append(accuracy)

            if phase == 'test' and accuracy > best_test_accuracy:
                best_model_weights = copy.deepcopy(model.state_dict())

        print(f'Ended at {datetime.now()}')
        print()

    torch.save(best_model_weights, (os.path.join(out_folder, 'model.pt')), 
               _use_new_zipfile_serialization=False)

    accuracy_pkl = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

    with open(os.path.join(out_folder, 'accuracy_list.pkl'), 'wb') as f:
        pickle.dump(accuracy_pkl, f)


if __name__ == '__main__':
    dataloader, dataset_size = prepare_dataset()

    loss_fn = nn.CrossEntropyLoss()
    
    # VGG-16 model with pre-trained weights
    # model = prepare_vgg_model(pretrained=True) # Load pretrained weights from PyTorch
    
    model = prepare_own_model()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=2,
        verbose=True)

    # out_folder = 'vgg_output'
    out_folder = 'own_model_output'

    train(model, dataloader, dataset_size, 
          optimizer, loss_fn, lr_scheduler, out_folder)
