#!/usr/bin/env python3
import sys
import subprocess

def run_mpileup(fasta, tsv, bam, pileup_out):
    """
    Runs `samtools mpileup` on the given BAM, with regions from TSV, writing stdout to `pileup_out` file,
    while streaming only samtools stderr (logs/messages) to the terminal.
    """
    cmd = [
        'samtools', 'mpileup',
        '-f', fasta,
        '-l', tsv,
        bam
    ]
    # Inform user of the command being executed
    print(f"Running: {' '.join(cmd)}")

    # Open output file for stdout
    with open(pileup_out, 'w') as out_f:
        # Start samtools, redirect stdout to file, capture stderr
        proc = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Stream stderr to terminal
        for err_line in proc.stderr:
            sys.stderr.write(err_line)

        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    fasta = "/cluster/work/boeva/minjwang/data/hg19/hg19.fa"

    bam_file = "/cluster/work/boeva/shoenig/ews-ml/data/A673_WT_fastq/HiCPro/bowtie_results/bwt2/sample1/sample1_merged.coord.bam"

    # Construct expected TSV path from previous step
    tsv_path = "/cluster/work/boeva/shoenig/ews-ml/data/ledidi/1mute.tsv"

    # Pileup output
    pileup_path = "/cluster/work/boeva/shoenig/ews-ml/src/ledidi/out.pileup"

    # Run samtools mpileup: stdout → pileup file; stderr → terminal
    run_mpileup(fasta, tsv_path, bam_file, pileup_path)


if __name__ == '__main__':
    main()
