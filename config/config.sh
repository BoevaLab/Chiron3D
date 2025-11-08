# config/config.sh

# HeLaS3
# C.Origami Model Paths
export WEIGHTS_HeLaS3_BASE="/cluster/work/boeva/shoenig/ews-ml/results/corigami/HeLaS3/25-03-BASE_EP26/epoch=26-step=119907.ckpt"
export WEIGHTS_HeLaS3_GATC="/cluster/work/boeva/shoenig/ews-ml/results/corigami/HeLaS3/06-04-GATC-1hot_EP22/epoch=22-step=102143.ckpt"
export WEIGHTS_HeLaS3_ATAT="/cluster/work/boeva/shoenig/ews-ml/results/corigami/HeLaS3/27-04-ATAT_EP18/epoch=18-step=84379.ckpt"
export WEIGHTS_HeLaS3_ZERO="/cluster/work/boeva/shoenig/ews-ml/results/corigami/HeLaS3/27-04-ZERO_EP16/epoch=16-step=75497.ckpt"

# Genomic Data
export COOL_HeLaS3="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/HeLaS3/contact_matrix_data/HeLaS3_CTCF_5000.cool"
export CTCF_HeLaS3="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/HeLaS3/genomic_features_data/GSE108869_Control_CTCF_ChIPSeq_treat_fc.bw"

# Loop calling file
export LOOPS_HeLaS3="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/HeLaS3/filtered_loops.csv"

# A673_WT
# C.Origami Model Paths
export WEIGHTS_A673_WT_BASE="/cluster/work/boeva/shoenig/ews-ml/results/corigami/A673_WT/BASE/epoch=27-step=124348.ckpt"
export WEIGHTS_A673_WT_GATC="/cluster/work/boeva/shoenig/ews-ml/results/corigami/A673_WT/GATC/epoch=24-step=111025.ckpt"
export WEIGHTS_A673_WT_ATAC="/cluster/work/boeva/shoenig/ews-ml/results/corigami/A673_WT/ATAC/epoch=23-step=106584.ckpt"
export WEIGHTS_A673_WT_DNA="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/Corigami-dna/models/epoch=2-step=1668.ckpt"

# Enformer Model Paths
export WEIGHTS_A673_WT_ENFORMER_SHALLOW="/cluster/work/boeva/shoenig/ews-ml/results/corigami/A673_WT/ENF-b/epoch=6-step=32403.ckpt"

#Borzoi Model Paths
export WEIGHTS_A673_WT_BORZOI_TFLORA="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/Borzoi-LoraTFLAYERS/models/epoch=12-step=14391.ckpt"
export WEIGHTS_A673_WT_BORZOI_FULLLORA="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/debug-lora-add-aug/models/epoch=14-step=16605.ckpt"
export WEIGHTS_A673_WT_BORZOI_CTCFLORA="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/Borzoi-LORA-CTCF/models/epoch=14-step=16605.ckpt"

export WEIGHTS_A673_WT_BORZOI_FULLLORA_STABLE="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/debug-lora-add-aug/models/epoch=14-step=16605.ckpt"

export WEIGHTS_A673_WT_BORZOI_TFLORA_45k="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/Borzoi-LoraTFLAYERS/models/epoch=12-step=14391.ckpt"
export WEIGHTS_A673_WT_BORZOI_FULLLORA_45k="/cluster/work/boeva/shoenig/ews-ml/training_runs_A673_WT/checkpoints/debug-lora-add-aug/models/epoch=14-step=16605.ckpt"
export WEIGHTS_A673_WT_BORZOI_CTCFLORA_45k="/cluster/work/boeva/shoenig/ews-ml/corigami_runs/checkpoints/Borzoi-LORA-CTCF/models/epoch=14-step=16605.ckpt"

# Genomic Data
export COOL_A673_WT="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/contact_matrix_data/A673_WT_CTCF_5000.cool"
export GENOM_FEAT_CTCF_ATAC_A673_WT="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/genomic_features_data/CTCF_ATAC"
export GENOM_FEAT_CTCF_A673_WT="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/genomic_features_data/CTCF"

# Loop calling file
export LOOPS_A673_WT="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/A673_WT/500kb_loops.csv"

# Stable Loop file
export STABLE_LOOPS_A673_WT="/cluster/work/boeva/shoenig/ews-ml/data/stable_extruding/stable_500kb_loops.csv"

# TC71
export COOL_TC71_WT="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/TC71_WT/contact_matrix_data/TC71_WT_CTCF_5000.cool"
export LOOPS_TC71_WT="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/processed/TC71_WT/500kb_loops.csv"
export WEIGHTS_TC71_WT_BORZOI_FULLLORA="/cluster/work/boeva/shoenig/ews-ml/training_runs_TC71_WT/checkpoints/Borzoi_FullLora2/models/epoch=7-step=8856.ckpt"

export WEIGHTS_TC71_WT_BORZOI_FULLLORA_STABLE="/cluster/work/boeva/shoenig/ews-ml/training_runs_TC71_WT/checkpoints/Borzoi_FullLora2/models/epoch=7-step=8856.ckpt"
export STABLE_LOOPS_TC71_WT="/cluster/work/boeva/shoenig/ews-ml/data/stable_extruding/stable_loops_TC71.csv"

export GENOM_FEAT_CTCF_TC71_WT="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/genomic_features_data/CTCF"
export GENOM_FEAT_CTCF_ATAC_TC71_WT="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/genomic_features_data/CTCF_ATAC"

export WEIGHTS_TC71_WT_BORZOI_FULLLORA2="/cluster/work/boeva/shoenig/ews-ml/training_runs_TC71_WT/checkpoints/Borzoi_FullLora/models/epoch=17-step=19926.ckpt"



# HG19 stuff
export FASTA_DIR_HG19="/cluster/work/boeva/minjwang/data/hg19/chromosomes"
export BLACKLIST_HG19="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/raw/HeLaS3/basenji_blacklist_hg19.bed"
export MOTIFS_MEME_FILE="/cluster/work/boeva/shoenig/ews-ml/data/loop_calling/raw/HeLaS3/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme"
export REGIONS_FILE_1MB="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/genomic_regions_data/filtered_hg19_windows.bed"
export REGIONS_FILE_200KB="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/genomic_regions_data/windows_hg19_enf.full.bed"
export REGIONS_FILE_500KB="/cluster/work/boeva/shoenig/ews-ml/data/corigami/raw/A673_WT/genomic_regions_data/windows_hg19_borzoi.full.bed"
