
import os
import pyfaidx
import torch
import numpy as np
from Bio import motifs
from torch import nn
from tqdm import tqdm
import os
import pyfaidx
from src.models.dataset.utils import onehotencode_dna

class SparseWriter:
    def __init__(self, chrom, bedgraph_fh):
        self.chrom = chrom
        self.bg = bedgraph_fh
        self.in_run = False
        self.run_start = None
        self.run_val = None

    def _emit(self, start, end, val):
        self.bg.write(f"{self.chrom}\t{start}\t{end}\t{val:.{2}f}\n")

    def consume(self, values: np.ndarray, start0: int):
        pos = start0
        for v in values:
            vq = round(float(v), 2)
            if vq > 0:
                if not self.in_run:
                    self.in_run = True
                    self.run_start = pos
                    self.run_val = vq
                elif vq != self.run_val:
                    self._emit(self.run_start, pos, self.run_val)
                    self.run_start = pos
                    self.run_val = vq
            else:
                if self.in_run:
                    self._emit(self.run_start, pos, self.run_val)
                    self.in_run = False
            pos += 1

    def flush(self, end_pos):
        if self.in_run:
            self._emit(self.run_start, end_pos, self.run_val)
            self.in_run = False


def discover_chroms_and_sizes(fasta_dir, prefer_order=None):
    """Find per-chrom FASTAs (*.fa) and return [(chrom, size, path_to_fa)]
       Ordered as 1..22, X, Y; extras afterward lexicographically.
    """
    fa_files = [f for f in os.listdir(fasta_dir) if f.endswith(".fa")]
    chrom_to_path = {os.path.splitext(f)[0]: os.path.join(fasta_dir, f) for f in fa_files}

    UCSC_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]
    order_index = {name: i for i, name in enumerate(UCSC_ORDER)}

    def core_name(chrom: str) -> str:
        return chrom[3:] if chrom.lower().startswith("chr") else chrom

    preferred = [c for c in chrom_to_path if core_name(c) in order_index]
    preferred.sort(key=lambda c: order_index[core_name(c)])
    extras = sorted([c for c in chrom_to_path if core_name(c) not in order_index])

    chroms = []
    print(preferred + extras)
    for chrom in preferred + extras:
        fapath = chrom_to_path[chrom]
        fa = pyfaidx.Fasta(fapath, as_raw=True, sequence_always_upper=True)
        try:
            size = len(fa[chrom])
        except KeyError:
            key = list(fa.keys())[0]
            size = len(fa[key])
        finally:
            fa.close()
        chroms.append((chrom, size, fapath))
    return chroms


def load_jaspar_pssm(jaspar_file):
    with open(jaspar_file) as fh:
        m = motifs.read(fh, "jaspar")
    
    pssm_fwd = m.pssm
    m_rc = m.reverse_complement()
    pssm_rev = m_rc.pssm

    def to_kernel(pssm_obj):
        K = pssm_obj.length
        mat = torch.full((4, K), -1e6, dtype=torch.float32)
        mat[0, :] = torch.tensor(pssm_obj['A'], dtype=torch.float32)
        mat[1, :] = torch.tensor(pssm_obj['C'], dtype=torch.float32)
        mat[2, :] = torch.tensor(pssm_obj['G'], dtype=torch.float32)
        mat[3, :] = torch.tensor(pssm_obj['T'], dtype=torch.float32)
        return mat.unsqueeze(0)  # [1,4,K]

    return to_kernel(pssm_fwd), to_kernel(pssm_rev)


def build_conv(kernel: torch.Tensor, device):
    k = int(kernel.size(-1))
    conv = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=k, stride=1, padding=0, bias=False)
    with torch.no_grad():
        conv.weight.copy_(kernel)
    for p in conv.parameters():
        p.requires_grad_(False)
    return conv.to(device)


@torch.no_grad()
def pwm_scan_scores(onehot_4xL: torch.Tensor, conv: torch.nn.Conv1d, device) -> torch.Tensor:
    x = onehot_4xL.unsqueeze(0).to(device)  # [1,4,L]
    s = conv(x).squeeze(0).squeeze(0).cpu() # [L-K+1]
    return s


def windows_to_perbase_max(signal_windows: torch.Tensor, L: int, K: int) -> torch.Tensor:
    s = signal_windows
    if len(s) == 0:
        return torch.zeros(L, dtype=torch.float32)
    tmp = torch.full((K, L), -float('inf'), dtype=torch.float32)
    for o in range(K):
        tmp[o, o:o+len(s)] = s
    perbase = tmp.max(dim=0).values
    perbase[~torch.isfinite(perbase)] = 0.0
    return perbase


def main(fasta_dir: str, jaspar_paths: dict, out_bedgraph: str, chunk_bp: int = 1_000_000):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load in all motif pwm
    motif_convs = []
    for name, path in jaspar_paths.items():
        k_fwd, k_rev = load_jaspar_pssm(path)
        conv_f = build_conv(k_fwd, device)
        conv_r = build_conv(k_rev, device)
        K = int(k_fwd.size(-1))
        motif_convs.append((name, conv_f, conv_r, K))

    max_K = max(K for _, _, _, K in motif_convs)
    chrom_infos = discover_chroms_and_sizes(fasta_dir)
    # Loop over sequence
    os.makedirs(os.path.dirname(out_bedgraph), exist_ok=True)
    with open(out_bedgraph, "w") as bg:
        for chrom, chrom_len, chrom_fa_path in chrom_infos:
            print(f"[{chrom}] len={chrom_len:,}")
            writer = SparseWriter(chrom, bedgraph_fh=bg)
            fa = pyfaidx.Fasta(chrom_fa_path, as_raw=True, sequence_always_upper=True)
            
            # Load in partsequence
            for s in tqdm(range(0, chrom_len, chunk_bp)):
                e = min(chrom_len, s + chunk_bp)

                padL = min(max_K - 1, s)
                padR = min(max_K - 1, chrom_len - e)
                qstart = s - padL
                qend   = e + padR

                try:
                    seq = str(fa[chrom][qstart:qend])
                except KeyError:
                    key = list(fa.keys())[0]
                    seq = str(fa[key][qstart:qend])

                Lfull = len(seq)
                central_len = e - s

                # one hot encode sequence
                oh = onehotencode_dna(seq, channels=4).to(torch.float32)
                if oh.shape[0] != 4:
                    raise ValueError("onehotencode_dna must return [4, L].")
                
                perbase_by_motif = []
                for _, conv_f, conv_r, K in motif_convs:
                    if Lfull < K:
                        continue
                    s_f = pwm_scan_scores(oh, conv_f, device)
                    s_r = pwm_scan_scores(oh, conv_r, device)
                    s_max = torch.maximum(s_f, s_r).clamp_min(0.0)
                    perbase = windows_to_perbase_max(s_max, Lfull, K)
                    perbase_by_motif.append(perbase)

                if perbase_by_motif:
                    combined = torch.stack(perbase_by_motif, dim=0).max(dim=0).values
                else:
                    combined = torch.zeros(Lfull, dtype=torch.float32)

                central = combined[padL:padL + central_len]
                writer.consume(central.numpy().astype(np.float32), s)
            
            writer.flush(chrom_len)
            fa.close()
    print("Done.")


if __name__ == "__main__":
    fasta_dir="/cluster/work/boeva/minjwang/data/hg19/chromosomes"
    """jaspar_paths={
        "SP1":  "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/MA0079.3.jaspar",
        "SP2":  "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/MA0516.1.jaspar",
        "SP3":  "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/MA0746.1.jaspar",
        "KLF3": "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/MA1516.1.jaspar",
    }"""

    jaspar_paths={
        "ERG":  "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/ERG.jaspar",
        "FLI1_1": "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/FLI1_1.jaspar",
        "FLI1_2": "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/FLI1_2.jaspar",
        "JUN_FOSB": "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/JUN_FOSB.jaspar",
        "JUN_FOSL2": "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/JUN_FOSL2.jaspar",
        "JUN": "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/JUN.jaspar",
        "JUN_JUNB": "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/JUN_JUNB.jaspar"
    }
    
    out_bedgraph="/cluster/work/boeva/shoenig/ews-ml/data/ledidi/fli1_jun.bedgraph"
    main(fasta_dir=fasta_dir, jaspar_paths=jaspar_paths, out_bedgraph=out_bedgraph)
