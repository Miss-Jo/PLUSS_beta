"""
Test script to verify multi-GPU setup and distributed training
"""

import torch
import torch.distributed as dist
import os


def test_gpu_availability():
    """Test GPU availability"""
    print("=" * 60)
    print("GPU Availability Test")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi Processors: {props.multi_processor_count}")
    else:
        print("No CUDA GPUs available!")
    
    print("=" * 60)


def test_distributed_init():
    """Test distributed initialization"""
    print("\n" + "=" * 60)
    print("Distributed Initialization Test")
    print("=" * 60)
    
    # Check environment variables
    env_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
    print("\nEnvironment Variables:")
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Try to initialize distributed
    if 'RANK' in os.environ:
        try:
            dist.init_process_group(backend='nccl')
            print(f"\n✓ Distributed initialized successfully!")
            print(f"  World Size: {dist.get_world_size()}")
            print(f"  Rank: {dist.get_rank()}")
            print(f"  Local Rank: {os.environ.get('LOCAL_RANK')}")
            
            # Test all-reduce
            tensor = torch.ones(1).cuda()
            dist.all_reduce(tensor)
            print(f"\n✓ All-reduce test passed!")
            print(f"  Result: {tensor.item()} (expected: {dist.get_world_size()})")
            
            dist.destroy_process_group()
        except Exception as e:
            print(f"\n✗ Distributed initialization failed: {e}")
    else:
        print("\n⚠ Not in distributed mode (RANK not set)")
        print("  To test distributed, run with:")
        print("  torchrun --nproc_per_node=2 test_multi_gpu.py")
    
    print("=" * 60)


def test_model_ddp():
    """Test DDP wrapping"""
    print("\n" + "=" * 60)
    print("DDP Model Wrapping Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, skipping DDP test")
        return
    
    if 'RANK' not in os.environ:
        print("⚠ Not in distributed mode, skipping DDP test")
        return
    
    try:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Create simple model
        model = torch.nn.Linear(10, 10).cuda(local_rank)
        
        # Wrap with DDP
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        
        # Forward pass
        input_tensor = torch.randn(4, 10).cuda(local_rank)
        output = model(input_tensor)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        print(f"✓ DDP test passed on rank {dist.get_rank()}")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        dist.destroy_process_group()
    except Exception as e:
        print(f"✗ DDP test failed: {e}")
    
    print("=" * 60)


def test_gradient_accumulation():
    """Test gradient accumulation"""
    print("\n" + "=" * 60)
    print("Gradient Accumulation Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, skipping test")
        return
    
    try:
        device = torch.device('cuda:0')
        model = torch.nn.Linear(10, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        accumulation_steps = 4
        
        print(f"Accumulation steps: {accumulation_steps}")
        
        # Simulate gradient accumulation
        optimizer.zero_grad()
        total_loss = 0
        
        for step in range(accumulation_steps):
            input_tensor = torch.randn(2, 10).to(device)
            target = torch.randn(2, 1).to(device)
            
            output = model(input_tensor)
            loss = torch.nn.functional.mse_loss(output, target)
            
            # Scale loss
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            total_loss += loss.item()
        
        # Update after accumulation
        optimizer.step()
        
        print(f"✓ Gradient accumulation test passed!")
        print(f"  Average loss: {total_loss / accumulation_steps:.4f}")
        
    except Exception as e:
        print(f"✗ Gradient accumulation test failed: {e}")
    
    print("=" * 60)


def test_mixed_precision():
    """Test mixed precision training"""
    print("\n" + "=" * 60)
    print("Mixed Precision Training Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, skipping test")
        return
    
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        device = torch.device('cuda:0')
        model = torch.nn.Linear(10, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()
        
        # Forward pass with autocast
        with autocast():
            input_tensor = torch.randn(4, 10).to(device)
            target = torch.randn(4, 1).to(device)
            output = model(input_tensor)
            loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✓ Mixed precision test passed!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Scaler scale: {scaler.get_scale()}")
        
    except Exception as e:
        print(f"✗ Mixed precision test failed: {e}")
    
    print("=" * 60)


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PLUSS_β Multi-GPU Environment Test")
    print("=" * 60)
    
    test_gpu_availability()
    test_distributed_init()
    test_model_ddp()
    test_gradient_accumulation()
    test_mixed_precision()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nRecommendations:")
    print("1. If GPUs are available but distributed fails:")
    print("   Run: torchrun --nproc_per_node=2 test_multi_gpu.py")
    print("\n2. If you see NCCL errors:")
    print("   export NCCL_DEBUG=INFO")
    print("   export NCCL_IB_DISABLE=1")
    print("\n3. For production training:")
    print("   bash pluss_beta/scripts/train_multi_gpu.sh")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
