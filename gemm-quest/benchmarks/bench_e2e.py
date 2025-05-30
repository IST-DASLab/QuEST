import json
import sys
sys.path.append('../../src')

from huggingface_hub import snapshot_download
from matplotlib import pyplot as plt, font_manager
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from fast_hadamard_transform import hadamard_transform

from models.quantization.base_linear import QuantizedLinear
from models.utils import get_model
from optim.utils import load_checkpoint

from utils import preprocess_weight, linear_forward, do_bench


def load_1600m():
    class DotDict(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class PseudoDdp(nn.Module):
        def __init__(self, model):
            super().__init__()
            self._orig_mod = nn.ModuleDict({"module": model})

    with open('./config_1600M.json') as f:
        config = json.load(f)

    model = PseudoDdp(get_model(DotDict(config)))._orig_mod['module']
    return model


class Linear4bit(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, use_sparse: bool = False, use_hadamard: bool = False, use_noop: bool = False):
        super().__init__()
        self.use_hadamard: bool = use_hadamard
        self.use_noop: bool = use_noop

        nn_linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.bias = nn_linear.bias

        weight = nn_linear.weight.data
        self.weight = torch.empty_like(weight, device=torch.device('meta'))
        w_int_packed, w_scale, w_add, w_meta_e = preprocess_weight(weight=weight, use_sparse=use_sparse, use_hadamard=use_hadamard, backend='triton')
        self.register_buffer('w_int_packed', w_int_packed)
        self.register_buffer('w_scale', w_scale)
        self.register_buffer('w_add', w_add)
        self.register_buffer('w_meta_e', w_meta_e)


    def forward(self, x) -> torch.Tensor:
        if not self.use_noop:
            a = x.contiguous().flatten(end_dim=-2)
            y = linear_forward(activation=a, w_int_packed=self.w_int_packed, w_scale=self.w_scale, w_add=self.w_add, w_meta_e=self.w_meta_e, out=None, use_hadamard=self.use_hadamard, backend='triton')
            y = y.unflatten(dim=0, sizes=x.shape[:-1])
        else:
            if self.use_hadamard:
                x: torch.Tensor = hadamard_transform(x=x.unflatten(dim=-1, sizes=(-1, 128)), scale=2. ** -3.5).flatten(start_dim=-2)
            y = torch.zeros(*x.shape[:-1], self.weight.size(0), dtype=x.dtype, device=x.device)
        if self.bias is not None:
            y += self.bias
        return y


def bench_e2e_time(model: nn.Module, batch_size: int = 8, seq_len: int = 512, mode: str = 'bf16', use_hadamard: bool = False):
    device = torch.device('cuda')
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    model.freqs_cis = model.freqs_cis.to(device=device)
    transformer = model.transformer

    layers = {}
    for name, module in transformer.named_modules():
        if isinstance(module, QuantizedLinear | nn.Linear | Linear4bit):
            layers[name] = module
    for name, module in layers.items():
        n_out, n_in = module.weight.shape
        # print(f'{name}: {n_out} x {n_in}')
        if mode == 'bf16':
            new_module = nn.Linear(n_in, n_out, bias=module.bias is not None, dtype=torch.bfloat16, device=device)
        elif mode == 'int4':
            new_module = Linear4bit(n_in, n_out, bias=module.bias is not None, dtype=torch.bfloat16, device=device, use_sparse=False, use_hadamard=use_hadamard, use_noop=False)
        elif mode == 'int4sp':
            new_module = Linear4bit(n_in, n_out, bias=module.bias is not None, dtype=torch.bfloat16, device=device, use_sparse=True, use_hadamard=use_hadamard, use_noop=False)
        elif mode == 'noop':
            new_module = Linear4bit(n_in, n_out, bias=module.bias is not None, dtype=torch.bfloat16, device=device, use_sparse=False, use_hadamard=use_hadamard, use_noop=True)
        else:
            raise NotImplementedError(mode)
        transformer.set_submodule(name, new_module)

    n_warmup = 5
    input_ids = torch.randint(0, 32_000, (batch_size, seq_len), dtype=torch.int64, device=device)
    f = lambda: model(idx=input_ids)['logits']
    tf = do_bench(f, n_iter=1, n_warmup=n_warmup)

    graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
    s: torch.cuda.Stream = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmup):
            f()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        output_logits = f()
    g = lambda: graph.replay()
    tg = do_bench(g, n_iter=1, n_warmup=n_warmup)

    input_ids_2 = torch.randint(0, 32_000, (batch_size, seq_len), dtype=torch.int64, device=device)
    input_ids.copy_(input_ids_2)
    g()
    # assert output_logits.equal(f()), 'CUDA Graph Incorrect!'

    # assert tf >= tg, 'CUDA Graph is slower!'
    return min(tf, tg)


def get_timing(batch_sizes,seq_len: int):
    model = load_1600m()
    print('Model Loaded.')

    runtime = np.zeros((len(batch_sizes), 4))
    for i, batch_size in enumerate(tqdm(batch_sizes)):
        t_base = bench_e2e_time(model, batch_size=batch_size, seq_len=seq_len, mode='bf16', use_hadamard=False)
        t_kernel = bench_e2e_time(model, batch_size=batch_size, seq_len=seq_len, mode='int4', use_hadamard=False)
        t_kernel_had = bench_e2e_time(model, batch_size=batch_size, seq_len=seq_len, mode='int4', use_hadamard=True)
        t_noop = bench_e2e_time(model, batch_size=batch_size, seq_len=seq_len, mode='noop', use_hadamard=False)
        runtime[i] = t_base, t_kernel, t_kernel_had, t_noop
    print(runtime.tolist())
    return runtime


def main():
    plot_colors = ['#3274a1', '#e1812c', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    batch_sizes = 2 ** np.arange(0, 4)
    seq_len = 1 * 512

    runtime = get_timing(batch_sizes, seq_len)
    # runtime = [[0.00820159912109375, 0.006977081298828125, 0.007503032684326172, 0.0025534629821777344], [0.014844894409179688, 0.009854555130004883, 0.01077580451965332, 0.004026174545288086], [0.028223276138305664, 0.01824808120727539, 0.019972562789916992, 0.008173942565917969], [0.061135053634643555, 0.04083442687988281, 0.04377317428588867, 0.02344965934753418]]

    runtime = np.asarray(runtime)

    t_base, t_kernel, t_kernel_had, t_noop = np.split(runtime, 4, axis=-1)

    speedups = t_base.flatten() / t_kernel.flatten()
    speedups_had = t_base.flatten() / t_kernel_had.flatten()


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7., 3.5))
    bar_width = 0.35

    x_indexes = range(len(batch_sizes))
    ax.bar(x_indexes, speedups, width=bar_width, color=plot_colors[0], label='NO HT', zorder=2)
    ax.bar([x + bar_width for x in x_indexes], speedups_had, width=bar_width, color=plot_colors[1], label='HT', zorder=2)
    ax.axhline(y=1., color='black', linestyle='--', linewidth=1)
    ax.set_xticks([x + bar_width / 2 for x in x_indexes])
    ax.set_xticklabels(batch_sizes, rotation=0., ha='center')
    ax.set_xlabel(f'Number of Sequences (Sequence Length = {seq_len})')
    ax.set_ylim(0., 2.)
    ax.set_yticks(np.arange(0, 2.1, .25))
    ax.set_yticklabels([f'{tick:.2f}' for tick in ax.get_yticks()])
    ax.set_ylabel('Speedup')
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='both', length=0)
    ax.legend(ncol=2, loc='upper center', framealpha=1.)
    ax.set_title(f'End-to-End Prefill Speedup INT4 vs BF16 on RTX 4090')
    ax.set_facecolor((1., 1., 1., 1.))
    fig.set_facecolor((1., 1., 1., 0.))
    fig.tight_layout()
    fig.show()

if __name__ == '__main__':
    main()
