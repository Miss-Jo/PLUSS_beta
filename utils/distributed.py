"""
Distributed Training Utilities for Multi-GPU Training
Supports DDP (DistributedDataParallel) with gradient accumulation
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(rank, world_size, backend='nccl'):
    """
    Initialize distributed training
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}] Distributed training initialized with world_size={world_size}")


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get current process rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get world size"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronization barrier"""
    if dist.is_initialized():
        dist.barrier()


def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary of tensors across all processes
    
    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average the values
        
    Returns:
        Reduced dictionary
    """
    if not dist.is_initialized():
        return input_dict
    
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


def gather_tensors(tensor):
    """
    Gather tensors from all processes
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of gathered tensors (only valid on rank 0)
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    
    # Gather tensor sizes first
    tensor_size = torch.tensor(tensor.size(), device=tensor.device)
    size_list = [torch.zeros_like(tensor_size) for _ in range(world_size)]
    dist.all_gather(size_list, tensor_size)
    
    # Gather tensors
    if get_rank() == 0:
        # Flatten and gather
        max_size = max([s[0].item() for s in size_list])
        tensor_list = []
        
        for i in range(world_size):
            if i == 0:
                tensor_list.append(tensor)
            else:
                tensor_list.append(torch.zeros(max_size, *tensor.size()[1:], 
                                              device=tensor.device, dtype=tensor.dtype))
        
        # Pad tensor to max size
        if tensor.size(0) < max_size:
            padded = torch.zeros(max_size, *tensor.size()[1:], 
                               device=tensor.device, dtype=tensor.dtype)
            padded[:tensor.size(0)] = tensor
        else:
            padded = tensor
        
        dist.gather(padded, gather_list=tensor_list, dst=0)
        
        # Trim to actual sizes
        tensor_list = [t[:size_list[i][0]] for i, t in enumerate(tensor_list)]
        return tensor_list
    else:
        # Pad and send
        max_size = max([s[0].item() for s in size_list])
        if tensor.size(0) < max_size:
            padded = torch.zeros(max_size, *tensor.size()[1:], 
                               device=tensor.device, dtype=tensor.dtype)
            padded[:tensor.size(0)] = tensor
        else:
            padded = tensor
        
        dist.gather(padded, dst=0)
        return []


class GradientAccumulator:
    """
    Handles gradient accumulation for larger effective batch sizes
    """
    
    def __init__(self, model, accumulation_steps=1):
        """
        Args:
            model: Model to accumulate gradients for
            accumulation_steps: Number of steps to accumulate
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_update(self):
        """Check if we should update parameters"""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self):
        """Increment step counter"""
        self.current_step += 1
    
    def reset(self):
        """Reset step counter"""
        self.current_step = 0
    
    def scale_loss(self, loss):
        """Scale loss by accumulation steps"""
        return loss / self.accumulation_steps


def convert_sync_batchnorm(model):
    """
    Convert BatchNorm to SyncBatchNorm for distributed training
    
    Args:
        model: Model to convert
        
    Returns:
        Converted model
    """
    if dist.is_initialized():
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def wrap_model_ddp(model, device_id, find_unused_parameters=False):
    """
    Wrap model with DistributedDataParallel
    
    Args:
        model: Model to wrap
        device_id: GPU device ID
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DDP-wrapped model
    """
    if dist.is_initialized():
        return DDP(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=find_unused_parameters
        )
    return model


class AverageMeter:
    """Computes and stores the average and current value (distributed-aware)"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def synchronize(self):
        """Synchronize across all processes"""
        if not dist.is_initialized():
            return
        
        t = torch.tensor([self.sum, self.count], dtype=torch.float32, device='cuda')
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count if self.count > 0 else 0


def save_checkpoint_distributed(state, filename, is_best=False):
    """
    Save checkpoint (only on main process)
    
    Args:
        state: State dict to save
        filename: Checkpoint filename
        is_best: Whether this is the best model
    """
    if is_main_process():
        torch.save(state, filename)
        if is_best:
            import shutil
            best_filename = filename.replace('.pth', '_best.pth')
            shutil.copyfile(filename, best_filename)


def load_checkpoint_distributed(filename, map_location=None):
    """
    Load checkpoint on all processes
    
    Args:
        filename: Checkpoint filename
        map_location: Device to map to
        
    Returns:
        Loaded checkpoint
    """
    # Ensure all processes wait for checkpoint to be saved
    barrier()
    
    if map_location is None:
        map_location = {'cuda:0': f'cuda:{get_rank()}'}
    
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def print_on_main(message):
    """Print only on main process"""
    if is_main_process():
        print(message)
