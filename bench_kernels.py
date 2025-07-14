import math
import torch
from torch.nn import functional as F
import numpy as np
import os

# import custom kernels
from llama.kernels import minimal
from llama.kernels import v1
from llama.kernels import minimal_v2
from llama.kernels import v2
from llama.kernels import fdm
from llama.kernels import fdm_splitkv
    
# Test configuration
batch_size = 6
n_head = 32
seq_len = 1
cache_len = 2048
head_embd = 128 # fixed for the custom kernels

def manual_attn(q, k, v, mask=None):
    """Standard manual attention implementation"""
    head_embd = q.shape[-1]
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_embd)
    if mask is not None:
        scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(q)
    output = torch.matmul(scores, v)
    return output

def cuda_benchmark_with_warmup(func, *args, warmup_runs=10, benchmark_runs=20):
    """
    Accurate GPU performance testing using CUDA Events
    
    Args:
        func: Function to test
        *args: Function arguments
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
    
    Returns:
        (average_time(ms), std_time(ms), function_output)
    """
    # Warmup phase
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = func(*args)
    torch.cuda.synchronize()
    
    # Benchmark phase
    times = []
    result = None
    
    for i in range(benchmark_runs):
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Start timing
        start_event.record()
        
        with torch.no_grad():
            result = func(*args)
        
        # End timing
        end_event.record()
        
        # Wait for GPU completion
        torch.cuda.synchronize()
        
        # Get elapsed time (milliseconds)
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    
    return np.mean(times), np.std(times), result

def calculate_attention_flops(batch_size, n_heads, seq_len, cache_len, head_dim):
    """
    Calculate theoretical FLOPS for attention computation
    """
    total_seq_len = seq_len + cache_len
    
    # Q @ K^T: (bs, n_heads, seq_len, head_dim) @ (bs, n_heads, head_dim, total_seq_len)
    qk_flops = batch_size * n_heads * seq_len * head_dim * total_seq_len
    
    # Softmax FLOPS are relatively small, simplified here
    softmax_flops = batch_size * n_heads * seq_len * total_seq_len * 3  # exp + sum + div
    
    # Attention @ V: (bs, n_heads, seq_len, total_seq_len) @ (bs, n_heads, total_seq_len, head_dim)
    av_flops = batch_size * n_heads * seq_len * total_seq_len * head_dim
    
    total_flops = qk_flops + softmax_flops + av_flops
    return total_flops

def comprehensive_attention_benchmark():
    """Comprehensive performance comparison of attention implementations"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    q = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, n_head, seq_len + cache_len, head_embd, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, n_head, seq_len + cache_len, head_embd, dtype=torch.float16, device=device)
    empty_mask = torch.empty(0, dtype=torch.float16, device=device)
    
    print(f"Test Configuration:")
    print(f"  batch_size={batch_size}, n_head={n_head}, seq_len={seq_len}")
    print(f"  cache_len={cache_len}, head_embd={head_embd}")
    print(f"  Device: {device}")
    print(f"  Data type: {q.dtype}")
    print("=" * 80)
    
    # Define all implementations to test
    implementations = [
        ("Manual Attention", lambda q, k, v, mask: manual_attn(q, k, v, None)),
        ("Minimal", lambda q, k, v, mask: minimal.forward(q, k, v, mask)),
        ("V1", lambda q, k, v, mask: v1.forward(q, k, v, mask)),
        ("Minimal V2", lambda q, k, v, mask: minimal_v2.forward(q, k, v, mask)),
        ("V2", lambda q, k, v, mask: v2.forward(q, k, v, mask)),
        ("FDM", lambda q, k, v, mask: fdm.forward(q, k, v, mask)),
        ("FDM SplitKV", lambda q, k, v, mask: fdm_splitkv.forward(q, k, v, mask)),
    ]
    
    # Calculate theoretical FLOPS
    theoretical_flops = calculate_attention_flops(batch_size, n_head, seq_len, cache_len, head_embd)
    print(f"Theoretical FLOPS: {theoretical_flops/1e9:.2f} GFLOP")
    print("=" * 80)
    
    results = []
    reference_output = None
    
    for name, impl_func in implementations:
        print(f"Testing {name}...")
        
        try:
            # Performance testing
            mean_time, std_time, output = cuda_benchmark_with_warmup(
                impl_func, q, k, v, empty_mask,
                warmup_runs=15, benchmark_runs=25
            )
            
            # Calculate FLOPS
            flops = theoretical_flops / (mean_time / 1000)  # Convert to seconds
            
            # Save reference output (usually use manual attention as reference)
            if reference_output is None:
                reference_output = output.clone()
            
            # Numerical correctness check
            max_diff = torch.max(torch.abs(output - reference_output)).item()
            is_correct = max_diff < 1e-2
            
            results.append({
                'name': name,
                'time_ms': mean_time,
                'time_std': std_time,
                'flops_gflops': flops / 1e9,
                'max_diff': max_diff,
                'is_correct': is_correct,
                'output': output
            })
            
            print(f"  ✓ Completed: {mean_time:.3f}±{std_time:.3f}ms, {flops/1e9:.2f} GFLOPS")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results.append({
                'name': name,
                'time_ms': float('inf'),
                'time_std': 0,
                'flops_gflops': 0,
                'max_diff': float('inf'),
                'is_correct': False,
                'error': str(e)
            })
    
    # Output detailed results
    print("\n" + "=" * 100)
    print("Detailed Performance Test Results:")
    print("=" * 100)
    print(f"{'Implementation':<15} {'Time(ms)':<12} {'Std Dev':<10} {'GFLOPS':<10} {'Speedup':<8} {'Numerical Error':<12} {'Correct':<8}")
    print("-" * 100)
    
    # Find the fastest correct implementation as baseline
    valid_results = [r for r in results if r['is_correct'] and r['time_ms'] != float('inf')]
    if valid_results:
        fastest_time = min(r['time_ms'] for r in valid_results)
    else:
        fastest_time = results[0]['time_ms']
    
    for result in results:
        if 'error' in result:
            print(f"{result['name']:<15} {'ERROR':<12} {'-':<10} {'-':<10} {'-':<8} {'-':<12} {'✗':<8}")
        else:
            speedup = fastest_time / result['time_ms'] if result['time_ms'] > 0 else 0
            correctness = '✓' if result['is_correct'] else '✗'
            
            print(f"{result['name']:<15} "
                  f"{result['time_ms']:.3f}±{result['time_std']:.2f} "
                  f"{result['flops_gflops']:.2f}     "
                  f"{speedup:.2f}x    "
                  f"{result['max_diff']:.2e}   "
                  f"{correctness:<8}")
    
    # Analyze best implementation
    print("\n" + "=" * 80)
    print("Performance Analysis Summary:")
    print("=" * 80)
    
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['time_ms'])
        worst_result = max(valid_results, key=lambda x: x['time_ms'])
        
        print(f"🏆 Fastest Implementation: {best_result['name']}")
        print(f"   Time: {best_result['time_ms']:.3f}ms")
        print(f"   Performance: {best_result['flops_gflops']:.2f} GFLOPS")
        
        print(f"\n📊 Performance Range:")
        print(f"   Fastest: {best_result['time_ms']:.3f}ms ({best_result['name']})")
        print(f"   Slowest: {worst_result['time_ms']:.3f}ms ({worst_result['name']})")
        print(f"   Gap: {worst_result['time_ms']/best_result['time_ms']:.2f}x")
        
        # Calculate improvement over manual attention
        manual_result = next((r for r in results if 'Manual' in r['name']), None)
        if manual_result and manual_result['is_correct']:
            improvement = manual_result['time_ms'] / best_result['time_ms']
            print(f"\n🚀 Maximum Improvement over Manual Attention: {improvement:.2f}x")
    
    # Numerical correctness summary
    print(f"\n🔍 Numerical Correctness Check:")
    correct_count = sum(1 for r in results if r.get('is_correct', False))
    total_count = len([r for r in results if 'error' not in r])
    print(f"   Passed: {correct_count}/{total_count}")
    
    for result in results:
        if not result.get('is_correct', False) and 'error' not in result:
            print(f"   ⚠️  {result['name']}: Numerical error too large ({result['max_diff']:.2e})")

if __name__ == "__main__":
    comprehensive_attention_benchmark()