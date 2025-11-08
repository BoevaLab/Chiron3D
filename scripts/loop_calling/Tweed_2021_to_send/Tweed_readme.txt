
# Step1 : launch foreach chromosomes the derivative & the enrichment method
bash ./scripts/Tweed_launch_2021.sh -s step1

# Step2 : merge results for enrichment method
bash ./scripts/Tweed_launch_2021.sh -s step2

# Step3: merge results between both methods (intersection)
bash ./scripts/Tweed_launch_2021.sh -s step3

perl ./scripts/HiChip_CTCF_enrichment_TAD_with_peaks_filter_20201123_bis.pl -i ./results/A673_WT_CTCF_5000/putative_tads_d15e10i0 -o ./results/A673_WT_CTCF_5000/filter_tads_20201123_bis_true_i0 -d 15 -f 2 -m 2 -r 5000
perl ./scripts/HiChip_CTCF_coverage_TAD_with_peaks_filter_20180612.pl -i ./results/A673_WT_CTCF_5000/putative_tads -o ./results/A673_WT_CTCF_5000/filter_tads_20201123_bis_true_i0 -d 15 -e 10 -m 2 -r 5000
perl ./scripts/HiChip_CTCF_coverage_TAD_merge_prediction_20190617.pl -i ./results/A673_WT_CTCF_5000/filter_tads_20201123_bis_true_i0_r5000_d15_e10_m2.bed -j  ./results/A673_WT_CTCF_5000/filter_tads_20201123_bis_true_i0_r5000_d15_fc2_m2.bed -o ./results/A673_WT_CTCF_5000/merged_tads_best_20201123_bis_true_i0_r5000_d15_e10_m2_fc2 -r 5000
