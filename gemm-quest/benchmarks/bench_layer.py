import sys

from matplotlib import pyplot as plt, font_manager
import numpy as np
import torch
from tqdm import tqdm


from utils import linear_forward, do_bench_network


def get_timing():
    batch_sizes = 2 ** np.arange(5, 15)
    hidden_sizes = [(6144, 2048), (2048, 2048), (11264, 2048), (2048, 5632), (12288, 4096), (4096, 4096), (22016, 4096), (4096, 11008)]
    runtime_dict = {}
    for N, K in tqdm(hidden_sizes):
        runtime_dict[N, K] = []
        for M in batch_sizes:
            dtype: torch.dtype = torch.bfloat16
            device: torch.device = torch.device('cuda')
            # activation = torch.empty(M, K, dtype=dtype, device=device)
            weight = torch.empty(N, K, dtype=dtype, device=device)
            w_scale = torch.empty(N, 1, dtype=dtype, device=device)
            w_int_packed = torch.empty(N, K // 2, dtype=torch.uint8, device=device)
            w_add = torch.empty(N, 1, dtype=torch.int32, device=device)
            a_scale = torch.empty(M, 1, dtype=dtype, device=device)
            a_int_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
            a_int_row_sum = torch.empty(M, 1, dtype=torch.int32, device=device)
            c = torch.empty(M, N, dtype=torch.int32, device=device)
            # out = torch.empty(M, N, dtype=dtype, device=device)

            t_base = do_bench_network(
                lambda a_in, a_out: torch.matmul(a_in, weight.t(), out=a_out),
                M, K, N,
                dtype=dtype,
                device=device,
            )
            t_kernel = do_bench_network(
                lambda a_in, a_out: linear_forward(activation=a_in, w_int_packed=w_int_packed, w_scale=w_scale, w_add=w_add, w_meta_e=None, out=a_out, _a_scale=a_scale, _a_int_packed=a_int_packed, _a_int_row_sum=a_int_row_sum, _c=c, use_hadamard=False, backend='triton'),
                M, K, N,
                dtype=dtype,
                device=device,
            )
            t_kernel_had = do_bench_network(
                lambda a_in, a_out: linear_forward(activation=a_in, w_int_packed=w_int_packed, w_scale=w_scale, w_add=w_add, w_meta_e=None, out=a_out, _a_scale=a_scale, _a_int_packed=a_int_packed, _a_int_row_sum=a_int_row_sum, _c=c, use_hadamard=True, backend='triton'),
                M, K, N,
                dtype=dtype,
                device=device,
            )

            runtime_dict[N, K].append((float(M), t_base, t_kernel, t_kernel_had))

    print(runtime_dict)
    return runtime_dict


def plot_speedup_line(runtime_dict: dict):
    plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, ((N, K), runtime) in enumerate(runtime_dict.items()):
        runtime = np.asarray(runtime)
        batch_sizes, t_base, t_kernel, t_kernel_had = np.split(runtime, runtime.shape[-1], axis=-1)
        batch_sizes = batch_sizes.flatten()
        t_base = t_base.flatten()
        t_kernel = t_kernel.flatten()
        t_kernel_had = t_kernel_had.flatten()
        plt.plot(t_base / t_kernel, label=f'{N}x{K}', color=plot_colors[i], marker='.', linestyle='-')
        plt.plot(t_base / t_kernel_had, color=plot_colors[i], marker='.', linestyle='--')
    plt.xticks(range(len(batch_sizes)), batch_sizes.astype(int))
    plt.xlabel('batch size')
    plt.ylim(0., 4.)
    plt.legend(title='weight shape')
    plt.title(f'INT4 vs BF16 Layer Speedup on RTX 4090')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'layer_speedup.png')
    plt.show()
    plt.clf()


def plot_speedup_bar(runtime_dict: dict):
    plot_colors = ['#3274a1', '#e1812c', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    layer_names = ['Q·K·V', 'O', 'Gate·Up', 'Down']
    batch_size = 4096

    layer_shapes = []
    speedups = []
    speedups_had = []
    for i, ((N, K), runtime) in enumerate(runtime_dict.items()):
        runtime = np.asarray(runtime)
        _, t_base, t_kernel, t_kernel_had = np.squeeze(runtime[runtime[:, 0] == batch_size])
        layer_shapes.append((N, K))
        speedups.append(t_base / t_kernel)
        speedups_had.append(t_base / t_kernel_had)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7., 3.5))
    bar_width = 0.35

    x_indexes = range(len(layer_shapes[:4]))
    ax0.bar(x_indexes, speedups[:4], width=bar_width, color=plot_colors[0], label='NO HT', zorder=2)
    ax0.bar([x + bar_width for x in x_indexes], speedups_had[:4], width=bar_width, color=plot_colors[1], label='HT', zorder=2)
    ax0.axhline(y=1., color='black', linestyle='--', linewidth=1)
    ax0.set_xticks([x + bar_width / 2 for x in x_indexes])
    ax0.set_xticklabels(layer_names, rotation=0., ha='center')
    ax0.set_ylim(0., 4.)
    ax0.set_yticks(np.arange(0, 4.1, .5))
    ax0.set_yticklabels(ax0.get_yticks())
    ax0.set_ylabel('Speedup')
    ax0.grid(axis='y')
    ax0.tick_params(axis='both', which='both', length=0)
    ax0.legend(loc='upper left', framealpha=1.)
    ax0.set_title('800M Model')
    ax0.set_facecolor((1., 1., 1., 1.))

    x_indexes = range(len(layer_shapes[4:]))
    ax1.bar(x_indexes, speedups[4:], width=bar_width, color=plot_colors[0], label='NO HT', zorder=2)
    ax1.bar([x + bar_width for x in x_indexes], speedups_had[4:], width=bar_width, color=plot_colors[1], label='HT', zorder=2)
    ax1.axhline(y=1., color='black', linestyle='--', linewidth=1)
    ax1.set_xticks([x + bar_width / 2 for x in x_indexes])
    ax1.set_xticklabels(layer_names, rotation=0., ha='center')
    ax1.set_ylim(0., 4.)
    ax1.set_yticks(np.arange(0, 4.1, .5))
    ax1.set_yticklabels(ax0.get_yticks())
    ax1.grid(axis='y')
    ax1.tick_params(axis='y', labelleft=False)
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_title('7B Model')
    ax1.set_facecolor((1., 1., 1., 1.))

    fig.suptitle(f'INT4 vs BF16 Layer Speedup for Batch Size = {batch_size} on RTX 4090')
    fig.set_facecolor((1., 1., 1., 0.))
    fig.tight_layout()
    fig.subplots_adjust(top=.84)
    fig.show()


def main():
    runtime_dict = get_timing()
    # runtime_dict = {(6144, 2048): [(32.0, 1.5664100646972657e-05, 3.68952751159668e-05, 3.836154937744141e-05), (64.0, 1.6295909881591796e-05, 3.703832626342773e-05, 3.8576126098632815e-05), (128.0, 3.1006336212158206e-05, 3.687143325805664e-05, 3.888607025146484e-05), (256.0, 4.991292953491211e-05, 3.752708435058594e-05, 4.062652587890625e-05), (512.0, 7.977485656738282e-05, 6.266832351684571e-05, 6.761550903320313e-05), (1024.0, 0.00015482902526855468, 9.266138076782226e-05, 0.00010137557983398437), (2048.0, 0.00029935836791992185, 0.00016422271728515624, 0.00018064975738525392), (4096.0, 0.0006081700325012207, 0.0003091335296630859, 0.0003448486328125), (8192.0, 0.001201319694519043, 0.0006205081939697265, 0.0006702065467834473), (16384.0, 0.0023813486099243165, 0.0012459158897399902, 0.0013506054878234864)], (2048, 2048): [(32.0, 7.796287536621094e-06, 3.598928451538086e-05, 3.726482391357422e-05), (64.0, 8.368492126464844e-06, 3.600120544433594e-05, 3.7562847137451175e-05), (128.0, 1.208782196044922e-05, 3.6549568176269534e-05, 3.859996795654297e-05), (256.0, 1.753568649291992e-05, 3.789663314819336e-05, 4.087686538696289e-05), (512.0, 2.8681755065917968e-05, 3.921985626220703e-05, 4.4178962707519534e-05), (1024.0, 5.295276641845703e-05, 4.1937828063964845e-05, 5.069971084594727e-05), (2048.0, 0.00010226964950561523, 7.492303848266602e-05, 9.136199951171875e-05), (4096.0, 0.00020072460174560546, 0.00013153553009033204, 0.00016342401504516603), (8192.0, 0.0004084944725036621, 0.00024529695510864256, 0.0003087043762207031), (16384.0, 0.0008088350296020508, 0.0004986405372619628, 0.0006472110748291015)], (11264, 2048): [(32.0, 2.2113323211669922e-05, 3.629922866821289e-05, 3.7729740142822266e-05), (64.0, 3.0446052551269532e-05, 3.652572631835937e-05, 3.8123130798339845e-05), (128.0, 4.5800209045410155e-05, 3.7157535552978514e-05, 3.917217254638672e-05), (256.0, 7.904767990112305e-05, 6.142854690551757e-05, 6.44683837890625e-05), (512.0, 0.00015201568603515626, 8.797645568847656e-05, 9.298324584960938e-05), (1024.0, 0.0002832174301147461, 0.0001560211181640625, 0.00016477108001708985), (2048.0, 0.0005573153495788575, 0.0002753615379333496, 0.0002919673919677734), (4096.0, 0.0011066675186157226, 0.000542140007019043, 0.0005684018135070801), (8192.0, 0.0021861791610717773, 0.001068115234375, 0.001109910011291504), (16384.0, 0.004347848892211914, 0.0021367907524108888, 0.0022353649139404295)], (2048, 5632): [(32.0, 1.5938282012939452e-05, 5.604028701782227e-05, 5.9604644775390625e-05), (64.0, 1.4662742614746094e-05, 5.621910095214844e-05, 6.0439109802246094e-05), (128.0, 2.3126602172851562e-05, 5.36799430847168e-05, 5.750656127929687e-05), (256.0, 4.469156265258789e-05, 5.459785461425781e-05, 6.107091903686523e-05), (512.0, 7.659196853637695e-05, 6.105899810791016e-05, 7.420778274536133e-05), (1024.0, 0.00014688968658447266, 7.35163688659668e-05, 9.654760360717773e-05), (2048.0, 0.000284576416015625, 0.00012165307998657227, 0.00016492605209350586), (4096.0, 0.0005589604377746582, 0.00023299455642700195, 0.0003341555595397949), (8192.0, 0.0011075377464294434, 0.0005045652389526368, 0.0007009744644165039), (16384.0, 0.002198648452758789, 0.0009745359420776367, 0.0013836860656738282)], (12288, 4096): [(32.0, 0.00012543201446533204, 4.609823226928711e-05, 4.763603210449219e-05), (64.0, 0.0001171112060546875, 4.801750183105469e-05, 5.0067901611328125e-05), (128.0, 0.00014537572860717773, 4.884004592895508e-05, 5.179643630981445e-05), (256.0, 0.00016628503799438478, 7.624626159667969e-05, 8.11457633972168e-05), (512.0, 0.0003058910369873047, 0.00011197328567504882, 0.00012063980102539062), (1024.0, 0.0006033182144165039, 0.00018783807754516602, 0.00020437240600585939), (2048.0, 0.0011951208114624023, 0.0003477931022644043, 0.0003804802894592285), (4096.0, 0.002378535270690918, 0.0006697654724121093, 0.0007165670394897461), (8192.0, 0.004733943939208984, 0.0013137340545654296, 0.0014173388481140137), (16384.0, 0.009439027309417725, 0.0026080846786499024, 0.002891051769256592)], (4096, 4096): [(32.0, 1.7464160919189453e-05, 4.756450653076172e-05, 4.897117614746094e-05), (64.0, 1.946687698364258e-05, 4.7588348388671874e-05, 4.9614906311035155e-05), (128.0, 3.40580940246582e-05, 4.6420097351074216e-05, 4.945993423461914e-05), (256.0, 5.319118499755859e-05, 4.891157150268555e-05, 5.3834915161132815e-05), (512.0, 0.00010161399841308593, 5.08427619934082e-05, 5.9592723846435544e-05), (1024.0, 0.00019941329956054687, 9.083747863769531e-05, 0.00010738372802734374), (2048.0, 0.0004003405570983887, 0.00015189647674560547, 0.00018385648727416992), (4096.0, 0.0008005023002624512, 0.00028160810470581057, 0.0003510355949401855), (8192.0, 0.001591646671295166, 0.0005546212196350098, 0.0006948709487915039), (16384.0, 0.0031703829765319822, 0.0011178255081176758, 0.001407921314239502)], (22016, 4096): [(32.0, 0.00020154714584350586, 6.946325302124024e-05, 7.120370864868164e-05), (64.0, 0.00022679567337036133, 6.996393203735351e-05, 7.195472717285157e-05), (128.0, 0.00023146867752075196, 7.390975952148438e-05, 7.68899917602539e-05), (256.0, 0.00029577016830444335, 0.00010907649993896484, 0.00011401176452636718), (512.0, 0.000551605224609375, 0.000180971622467041, 0.00019037723541259766), (1024.0, 0.001090085506439209, 0.0003067612648010254, 0.000323641300201416), (2048.0, 0.0021363973617553713, 0.000573277473449707, 0.0005970239639282226), (4096.0, 0.004250955581665039, 0.0010898828506469727, 0.001132059097290039), (8192.0, 0.008463811874389649, 0.002158379554748535, 0.0022527217864990235), (16384.0, 0.01689506769180298, 0.004287993907928467, 0.0045724987983703615)], (4096, 11008): [(32.0, 0.00010818243026733398, 8.498430252075195e-05, 9.09566879272461e-05), (64.0, 0.00011299848556518554, 8.52823257446289e-05, 9.248256683349609e-05), (128.0, 0.00011371374130249023, 8.431673049926757e-05, 9.052753448486328e-05), (256.0, 0.00014685392379760743, 8.178949356079102e-05, 9.325742721557617e-05), (512.0, 0.0002791881561279297, 9.700059890747071e-05, 0.00012192726135253906), (1024.0, 0.0005466699600219727, 0.00016241073608398439, 0.00020751953125), (2048.0, 0.0010801434516906738, 0.0002954959869384766, 0.000379478931427002), (4096.0, 0.002150881290435791, 0.0006214141845703125, 0.0008040070533752441), (8192.0, 0.004290878772735596, 0.0011992573738098145, 0.00159759521484375), (16384.0, 0.008563745021820068, 0.002915811538696289, 0.0037134647369384765)]}

    plot_speedup_bar(runtime_dict)

if __name__ == '__main__':
    main()
