import numpy as np
import os
import torch
from torch.optim.lr_scheduler import OneCycleLR
import json

optimizer_dict = {
    "Adam":torch.optim.Adam,
    "SGD":torch.optim.SGD,
    "LBFGS":torch.optim.LBFGS
}

def save_data(file_path, x, y_new,epoch,header):

    #File does already exist
    if os.path.exists(file_path):
        existing_data = np.loadtxt(file_path, skiprows=1)
        # Add the epoch to the data
        y_new = np.concatenate((np.array([epoch]),y_new),axis=0)

        if (existing_data.shape[0] != len(x)+1) or (existing_data.shape[0] != len(y_new)):
            raise ValueError("The length of the x-values does not match the existing data.")
        
        updated_data = np.hstack((existing_data, y_new.reshape(-1, 1)))
        updated_data[1:,0] = x

        np.savetxt(file_path, updated_data, delimiter="\t",header = header)

    #Initial call
    else:
        # Add the epoch to the data
        x = np.concatenate((np.array([-1]),x),axis=0)
        y_new = np.concatenate((np.array([epoch]),y_new),axis=0)

        updated_data = np.hstack((x.reshape(-1, 1), y_new.reshape(-1, 1)))

        np.savetxt(file_path, updated_data, delimiter="\t",header = header)

class MultiCycleLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lrs, epochs_per_cycle, n_cycles=3, steps_per_epoch=None, div_factor=25, final_div_factor=10000, last_epoch=-1):
        assert len(max_lrs) == n_cycles, "The length of max_lrs should match the number of cycles."
        assert len(epochs_per_cycle) == n_cycles, "The length of epochs_per_cycle should match the number of cycles."
        
        self.max_lrs = max_lrs
        self.epochs_per_cycle = epochs_per_cycle
        self.n_cycles = n_cycles
        self.steps_per_epoch = steps_per_epoch
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.current_cycle = 0
        self.cycle_length = epochs_per_cycle[self.current_cycle] * steps_per_epoch
        self.one_cycle_lr = OneCycleLR(
            optimizer, max_lr=self.max_lrs[self.current_cycle], total_steps=self.cycle_length, 
            div_factor=self.div_factor, final_div_factor=self.final_div_factor
        )
        
        super(MultiCycleLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= sum([ep * self.steps_per_epoch for ep in self.epochs_per_cycle[:self.current_cycle + 1]]):
            self.current_cycle += 1
            if self.current_cycle < self.n_cycles:
                self.cycle_length = self.epochs_per_cycle[self.current_cycle] * self.steps_per_epoch
                self.one_cycle_lr = OneCycleLR(
                    self.optimizer, max_lr=self.max_lrs[self.current_cycle], total_steps=self.cycle_length, 
                    div_factor=self.div_factor, final_div_factor=self.final_div_factor
                )
        
        return self.one_cycle_lr.get_last_lr()
    
    def _last_lr(self):
        return self.get_lr()
    
    def get_last_lr(self):
        return self._last_lr()
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            epoch = epoch - self.base_lrs[0]
        
        self.last_epoch = epoch
        
        if self.current_cycle < self.n_cycles:
            self.one_cycle_lr.step(epoch - sum([ep * self.steps_per_epoch for ep in self.epochs_per_cycle[:self.current_cycle]]))

def remove_non_serializable(obj):
    """Recursively remove all non-JSON-serializable entries from a nested dictionary or list."""
    if isinstance(obj, dict):
        return {k: remove_non_serializable(v) for k, v in obj.items() if is_json_serializable(v)}
    elif isinstance(obj, list):
        return [remove_non_serializable(v) for v in obj if is_json_serializable(v)]
    else:
        return obj if is_json_serializable(obj) else None  # Optional: Remove or replace with None

def is_json_serializable(value):
    """Check if a value is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False
