import time
import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from utils_dataset import *
from utils_metrics import *
from utils import *

#####################################################
########## TRAINING

def full_train(model, optimizer, num_epochs, train_loader, val_loader, num_eval, 
               metrics_names, device, model_name = "model.pt", save = False, grad_masks = None):
    
    model.to(device)
    model.train()
    
    for epochs in range(num_epochs):
        
        print(f"Epoch [{epochs+1}/{num_epochs}] ", end = '')
        start = time.perf_counter()
        
        _ = train(model, optimizer, train_loader, val_loader, num_eval, metrics_names, device, model_name, save, grad_masks)
        
        end = time.perf_counter()
        print("Train Loss: {:.4f} Validation Loss: {:.4f} ---- Time: {:.4f} seconds".format(model.metrics.loss_train[-1],
                                                                                            model.metrics.loss_val[-1], end-start))
        
        for metric in metrics_names:
            print("{}: {:.4f}".format(metric, model.metrics.metrics[metric][-1]))
            
        model.metrics.times.append(end-start)
        model.metrics.epoch += 1
    
    model.metrics.print()

    

def train(model, optimizer, train_loader, val_loader, num_eval, metrics_names, device, model_name, save, grad_masks = None):
    
    model.train()
    print('|',end='')
    losses = []

    nb_evaluation = 0 # to be sure we make nb_eval evaluations

    for i, (encoding, labels) in enumerate(train_loader):
        
        # Get the output
        input_ids = encoding['input_ids'].to(device)
        attention_masks = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)
        labels = labels.to(device)
        
        # Optimize
        optimizer.zero_grad()
        output = model(input_ids, attention_mask=attention_masks, token_type_ids = token_type_ids, labels=labels)
        loss = output.loss
        loss.backward()

        # For random finetuning, we want to drop gradients we shall not use
        if grad_masks is not None:
          for n,p in model.named_parameters():
            p.grad *= grad_masks[n].to(device)

        optimizer.step()
        
        # Save the loss
        losses.append(loss.item())
        
        # My homemade load bar
        if ((i+1)%(max([int(len(train_loader)/50), 1]))==0):
            print('=', end = '')
            
        # Evaluate on the validation set num_eval times
        # We evaluate all metrics in model.metrics and the loss
        # We also save the training loss right now
        if ((i+1)%(int(len(train_loader)/num_eval))==0) and (nb_evaluation < num_eval):
            val_loss, _ = evaluate(model, val_loader, metrics_names, device, True)
            model.metrics.loss_val.append(val_loss)
            model.metrics.steps.append(model.metrics.epoch*len(train_loader) + i+1)
            nb_evaluation += 1

            model.metrics.loss_train.append(sum(losses)/len(losses))
            losses = []
            
            if len(model.metrics.steps)>0:
                if save & (model.metrics.metrics[list(model.metrics.metrics.keys())[0]][-1]==max(model.metrics.metrics[list(model.metrics.metrics.keys())[0]])):
                    save_model(model, model_name)
    
    print('|')
    model.eval()

def evaluate(model, dataloader, metrics_names, device, append):
    
    model.eval()
    losses = []
    metrics = {}
    
    preds = []
    targets = []

    with torch.no_grad():
        for (encoding, labels) in dataloader:

            input_ids = encoding['input_ids'].to(device)
            attention_masks = encoding['attention_mask'].to(device)
            token_type_ids = encoding['token_type_ids'].to(device)
            labels = labels.to(device)

            output = model(input_ids, attention_mask=attention_masks, token_type_ids = token_type_ids, labels=labels)
            loss = output.loss

            losses.append(loss.item())
            preds.append(torch.argmax(output.logits, dim = -1))
            targets.append(labels)
        
    preds = torch.cat(preds, dim = 0)
    targets = torch.cat(targets, dim = 0)
    for metric in metrics_names:
        metrics[metric] = compute_metric(preds, targets, metric)
        if append:
            model.metrics.metrics[metric].append(metrics[metric])

    model.train()
        
    return sum(losses)/len(losses), metrics

def compute_metric(preds, target, metric_name):
    """
    Given predictions and targets, gives the result for a desired metric
    preds (torch.tensor): predictions
    target (torch.tensor): targets
    metric_name (string): can  be 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'MatthewCorr'
    """
    
    metric = None
    
    if metric_name == 'F1 Score':
        metric = f1_score(target.cpu().numpy(), preds.cpu().numpy(), average = 'weighted')
    if metric_name == 'Accuracy':
        metric = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
    if metric_name == 'Precision':
        metric = precision_score(target.cpu().numpy(), preds.cpu().numpy(), average = 'weighted')
    if metric_name == 'Recall':
        metric = recall_score(target.cpu().numpy(), preds.cpu().numpy(), average = 'weighted')
    if metric_name == 'MatthewCorr':
        metric = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
        
    return metric

def training(hyperparameters):

    ############################################################################

    device = hyperparameters['device']
    metrics_names = hyperparameters['metrics_names']
    num_eval = hyperparameters['num_eval']
    model_name = hyperparameters['model_name']
    dataset_name = hyperparameters['dataset_name']
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']
    TRAIN_PATH = hyperparameters['TRAIN_PATH']
    TEST_PATH = hyperparameters['TEST_PATH']
    num_epochs = hyperparameters['num_epochs']
    seed = hyperparameters['seed']
    Finetuning = hyperparameters['Finetuning']
    subsample_train_size = hyperparameters['subsample_train_size']
    subsample_test_size = hyperparameters['subsample_test_size']
    max_length = hyperparameters['max_length']
    train_frac = hyperparameters['train_frac']
    grad_masks = hyperparameters['grad_masks']
    ratio_params = hyperparameters['ratio_params']

    ############################################################################

    torch.manual_seed(seed)
    np.random.seed(seed)

    if Finetuning == 'Random':
        save_name = f"Random{ratio_params}"
    else:
        save_name = f"{Finetuning}".replace('&', '_')
    
    if subsample_train_size != None:
        save_name += f"_Size{subsample_train_size}"
    
    save_name += f"_{model_name}_{dataset_name}_seed{seed}_lr{lr}_epochs{num_epochs}"

    print(f"\n==================================================================\n")
    print(f"{save_name}:\n\n")

    if (save_name+'.pt' not in os.listdir('Results/Models/'+dataset_name)) & (save_name+".pt" not in os.listdir()):


        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        model.metrics = Metrics(metrics_names, num_eval)
        model.full_losses = []
        
        # Define which parameters to tune depending on the Finetuning method
        if Finetuning == 'Full':
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        elif Finetuning == 'BitFit':
            params_bias = [
                        {'params': [p for n, p in model.named_parameters() if (('bias' in n) or ('classifier' in n))]}
            ]
            optimizer = torch.optim.Adam(params_bias, lr = lr)

        elif Finetuning == 'LayerNorm':
            params = [
            {'params': [p for n, p in model.named_parameters() if ('classifier' in n) or ('LayerNorm' in n)]}
            ]
            optimizer = torch.optim.Adam(params, lr = lr)
        
        elif Finetuning == 'Random':
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        elif Finetuning == "Init&BitFit":
            model.init_weights()
            params_bias = [
                    {'params': [p for n, p in model.named_parameters() if (('bias' in n) or ('classifier' in n))]}
            ]
            optimizer = torch.optim.Adam(params_bias, lr = lr)

        elif Finetuning == 'InitBias&BitFit':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    nn.init.zeros_(p)
            params_bias = [
                    {'params': [p for n, p in model.named_parameters() if (('bias' in n) or ('classifier' in n))]}
            ]
            optimizer = torch.optim.Adam(params_bias, lr = lr)

        # Recover the dataloaders
        train_loader, val_loader, test_loader = get_dataloader(dataset_name, batch_size, tokenizer,
                                                              TRAIN_PATH, TEST_PATH, train_frac,
                                                              subsample_train_size, subsample_test_size, max_length )

        # Train
        full_train(model, optimizer, num_epochs, train_loader, val_loader, num_eval, metrics_names,
                  device, model_name='Best_'+save_name, save = True, grad_masks = grad_masks)
    
        saved_model = torch.load('Best_'+save_name)

        model.load_state_dict(saved_model['state_dict'])

        test_loss, test_metrics = evaluate(model, test_loader, metrics_names, device, append = False)

        # We don't want to save the whole grad_masks, it is as big as the model parameters
        # Instead we only save the number of non-zeros per layer
        if grad_masks is not None:
            for key in grad_masks.keys():
                hyperparameters['grad_masks'][key] = torch.sum(grad_masks[key]).item()

        torch.save({'hyperparameters': hyperparameters,
                    'metrics': model.metrics,
                    'test_loss': test_loss,
                    'test_metrics': test_metrics},
                    save_name + ".pt")
        
        os.remove('Best_'+save_name)
        
    else:
        print(f"{save_name} already trained\n\n")