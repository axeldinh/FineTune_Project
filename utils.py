import torch
import numpy as np

###############################################################
####### Gadgets

def get_grad_mask(model, ratio_params):
    """
    Select random parameters for finetuning
    Input:
      -model: model from which we select the parameters
      -ratio_params: ratio of the total number of parameters we want
    Output:
      -grad_masks (dict): Dictionary of tensors in {0,1} in the same shape as in the model,
                          e.g {'param_name1': tensor1, 'param_name2': tensor2}
    """
  
    grad_masks = {}
    parameters_sizes = {}
    total_parameters = 0
    avail_parameters = 0
    classifier_params = 0
    for n, p in model.named_parameters():

      total_parameters += torch.numel(p)

      if 'classifier' not in n:
        parameters_sizes[n] = (p.size(), torch.numel(p))
        avail_parameters += torch.numel(p)
      else:
        classifier_params += torch.numel(p)
        grad_masks[n] = torch.ones(p.size())
        
    flatten_grad_mask = torch.zeros(avail_parameters)

    ft_parameters = int(ratio_params * total_parameters) - classifier_params

    i = 0

    while i < ft_parameters:

      idx = np.random.randint(avail_parameters)

      if flatten_grad_mask[idx] == 0:
        flatten_grad_mask[idx] = 1
        i += 1
      
    i = 0
    for n, p in model.named_parameters():

      if 'classifier' not in n:
        grad_masks[n] = flatten_grad_mask[i:i+parameters_sizes[n][1]].view(parameters_sizes[n][0])
        i += parameters_sizes[n][1]

    return grad_masks

def seconds_to_string(seconds):
    """
    Transforms the seconds in a string such as "3h34mn46s" for 3 hours 34 minutes and 46 seconds.
    Parameters:
        -seconds (int): number of seconds to convert
    """
    seconds = int(seconds)
    hours = int(seconds//(60**2))
    minutes = int(seconds%(60**2)) // 60
    seconds = int(seconds%(60**2)) % 60
    return f"{hours}h{minutes}mn{seconds}s"

def save_model(model, model_name):
    torch.save({'state_dict': model.state_dict(),
                'metrics': model.metrics}, model_name)
    
    