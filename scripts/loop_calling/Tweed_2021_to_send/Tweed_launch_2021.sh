#!/bin/bash

## RECUP OPTIONS
while getopts "s:" optionName; do
case "$optionName" in
    s) STEP="$OPTARG";;
esac
done

# PBS OPTIONS
##############

WORKDIR=/cluster/work/boeva/shoenig/ews-ml/src/loop_calling
TWEED_DIR=/cluster/work/boeva/shoenig/ews-ml/src/loop_calling/Tweed_2021_to_send

SAMPLE_HICHIP="HeLaS3_CTCF"

REF_PEAKS=/cluster/work/boeva/shoenig/ews-ml/src/loop_calling/data/GSE108869_Control_CTCF_ChIPSeq.narrowPeak

RESOLUTION=5000
OUTDIR=${WORKDIR}/results/
DELTA_DIAGONALE=15
DELTA_PEAKS=10
MIN_INTENSITY=0
OUTDIR="${OUTDIR}/${SAMPLE_HICHIP}_${RESOLUTION}/"
	
PBS_OUT=${WORKDIR}/pbsfiles/
TMPDIR=${WORKDIR}/tmp/
MEMORY=20gb
NBPROC=1
WALLTIME=3-00:00:00
JOBNAME="${STEP}"

if [ ! -e ${PBS_OUT} ]; then
	mkdir -p $PBS_OUT
fi  

if [ "${STEP}" == "step1" ]; then
	MEMORY=20gb
	WALLTIME=3-00:00:00

	POSITION_MATRIX=/cluster/work/boeva/shoenig/ews-ml/src/loop_calling/data/GSE108869_HeLaS3_POSITION_MATRIX.bed
	COVERAGE_MATRIX=/cluster/work/boeva/shoenig/ews-ml/src/loop_calling/data/GSE108869_HeLaS3_COVERAGE_MATRIX.matrix
	START_LIST="0 50000000 100000000 150000000 200000000"
	DELTA=55000000
	
	if [ -e ${REF_PEAKS} ]; then
		if [ ! -e ${OUTDIR} ]; then
			mkdir -p ${OUTDIR}
		fi
		if [ ! -e ${POSITION_MATRIX} ] ||  [ ! -e ${COVERAGE_MATRIX} ]; then
			echo "Missing file ${POSITION_MATRIX} and/or ${COVERAGE_MATRIX}"
		else
			for CHR in {1..24}; do
				if [ "${CHR}" == 23 ]; then
					CHR=chrX
				elif [ "${CHR}" == 24 ]; then
					CHR=chrY
				else
					CHR="chr"${CHR}
				fi

				OUTFILE=${OUTDIR}/putative_tads
				# Derivative method
				if [ ! -e ${OUTFILE}_${CHR}_derivative.out ]; then
					JOBNAME="${STEP}_DERIV_${SAMPLE_HICHIP}_${CHR}"
					COMMAND="#!/bin/bash\nmodule load perl\nmodule load bedtools\nexport PERL5LIB=\${PERL5LIB}:/cluster/customapps/biomed/boeva/shoenig/perl5/lib/perl5\nperl ${TWEED_DIR}/HiChip_CTCF_coverage_TAD_with_peaks_derivative_20180221.pl -i ${POSITION_MATRIX} -j  ${COVERAGE_MATRIX}  -p ${REF_PEAKS} -c ${CHR} -o ${OUTFILE} -r ${RESOLUTION} -t ${TMPDIR} -m ${MIN_INTENSITY}  "
					echo -e "${COMMAND}\n${PBS_OUT}/job_${JOBNAME}\n"	
					#JOB=`echo -e "${COMMAND}\n" | qsub -e ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.err -m ae -o ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.out -N ${JOBNAME} -q batch -l nodes=1:ppn=${NBPROC},mem=${MEMORY},walltime=${WALLTIME}`
					JOB=$(echo -e "${COMMAND}\n" | sbatch \
            --error=${PBS_OUT}/job_${JOBNAME}.%j.err \
            --output=${PBS_OUT}/job_${JOBNAME}.%j.out \
            --job-name=${JOBNAME} \
            --nodes=1 \
            --ntasks-per-node=${NBPROC} \
            --mem=${MEMORY} \
            --time=${WALLTIME})
				fi
				# Enrichment method : split file to go fast

				MAX_CHR=`awk -F"\t" '$1=="'${CHR}'" {print $3}' ${POSITION_MATRIX} | tail -n 1`
				I=1
				for START in ${START_LIST}; do
					if [ ${START}  -le ${MAX_CHR} ]; then
						END=$((START+DELTA))
						# OUTFILE=${OUTDIR}/putative_tads_20201118_d${DELTA_DIAGONALE}e${DELTA_PEAKS}i${MIN_INTENSITY}_part${I}
						OUTFILE=${OUTDIR}/putative_tads_d${DELTA_DIAGONALE}e${DELTA_PEAKS}i${MIN_INTENSITY}_part${I}
						if [ ! -e ${OUTFILE}_${CHR}_enrichment.out ]; then
							COMMAND="#!/bin/bash\nmodule load perl\nmodule load bedtools\nperl ${TWEED_DIR}/HiChip_CTCF_coverage_TAD_with_peaks_line_enrichment_20201118_split.pl -i ${POSITION_MATRIX} -j  ${COVERAGE_MATRIX}  -p ${REF_PEAKS} -c ${CHR} -o ${OUTFILE} -m ${MIN_INTENSITY} -d ${DELTA_DIAGONALE} -e ${DELTA_PEAKS} -a ${START} -b ${END}"
							JOBNAME="${STEP}_d${DELTA_DIAGONALE}e${DELTA_PEAKS}i${MIN_INTENSITY}r${RESOLUTION}_ENRICH_${SAMPLE_HICHIP}_${CHR}_${I}"
							echo -e "${COMMAND}\n${PBS_OUT}/job_${JOBNAME}\n"	# exit 0
							#JOB=`echo -e "${COMMAND}\n" | qsub -e ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.err -m ae -o ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.out -N ${JOBNAME}   -q batch -l nodes=1:ppn=${NBPROC},mem=${MEMORY},walltime=${WALLTIME}`
							JOB=$(echo -e "${COMMAND}\n" | sbatch \
                --error=${PBS_OUT}/job_${JOBNAME}.%j.err \
                --output=${PBS_OUT}/job_${JOBNAME}.%j.out \
                --job-name=${JOBNAME} \
                --nodes=1 \
                --ntasks-per-node=${NBPROC} \
                --mem=${MEMORY} \
                --time=${WALLTIME})
						else
							echo "${OUTFILE}_${CHR}_enrichment.out  already exists"
						fi
					fi
					I=$((I+1))
				done
			done			
		fi
	else
		echo "No file ${REF_PEAKS}"
	fi
elif [ "${STEP}" == "step2" ]; then
	MEMORY=20gb
	WALLTIME=3-00:00:00

	COMMAND="#!/bin/bash\nmodule load perl\nmodule load bedtools\nperl ${TWEED_DIR}/HiChip_CTCF_enrichment_TAD_with_peaks_merge_20200529.pl -i ${OUTDIR} -f putative_tads_d${DELTA_DIAGONALE}e${DELTA_PEAKS}i${MIN_INTENSITY}"
	JOBNAME="${STEP}_${SAMPLE_HICHIP}"							
	echo -e "${COMMAND}\n${PBS_OUT}/job_${JOBNAME}\n"	# exit 0
	#JOB=`echo -e "${COMMAND}\n" | qsub -e ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.err -m ae -o ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.out -N ${JOBNAME}   -q batch -l nodes=1:ppn=${NBPROC},mem=${MEMORY},walltime=${WALLTIME}`
	JOB=$(echo -e "${COMMAND}\n" | sbatch \
    --error=${PBS_OUT}/job_${JOBNAME}.%j.err \
    --output=${PBS_OUT}/job_${JOBNAME}.%j.out \
    --job-name=${JOBNAME} \
    --nodes=1 \
    --ntasks-per-node=${NBPROC} \
    --mem=${MEMORY} \
    --time=${WALLTIME})

elif [ "${STEP}" == "step3" ]; then
	MEMORY=5gb
	DELTA_DIAGONALE=15
	DELTA_REF=10
	FC=2 
	MIN_READS_BIN=2

	INFILE=${OUTDIR}/putative_tads
	for CHR in {1..24}; do
		if [ "${CHR}" == 23 ]; then
			CHR=chrX
		elif [ "${CHR}" == 24 ]; then
			CHR=chrY
		else
			CHR="chr"${CHR}
		fi
		if [ ! -e ${INFILE}_${CHR}_derivative.out ]; then
			echo "No infile ${INFILE}_${CHR}_derivative.out"
			exit 1
		fi
	done
		JOBNAME="${STEP}_${SAMPLE_HICHIP}"
		OUTFILE=${OUTDIR}/filter_tads_20201123_bis_true_i${MIN_INTENSITY}
		OUTFILE_DERIV=${OUTFILE}_r${RESOLUTION}_d${DELTA_DIAGONALE}_e${DELTA_REF}_m${MIN_READS_BIN}.bed
		OUTFILE_ENRICHED=${OUTFILE}_r${RESOLUTION}_d${DELTA_DIAGONALE}_fc${FC}_m${MIN_READS_BIN}.bed
		OUTFILE_MERGED=${OUTDIR}/merged_tads_best_20201123_bis_true_i${MIN_INTENSITY}_r${RESOLUTION}_d${DELTA_DIAGONALE}_e${DELTA_REF}_m${MIN_READS_BIN}_fc${FC}
		if [ ! -e ${OUTFILE_MERGED}.juicebox ]; then 
			COMMAND="#!/bin/bash\nmodule load perl\nmodule load bedtools\nperl ${TWEED_DIR}/HiChip_CTCF_enrichment_TAD_with_peaks_filter_20201123_bis.pl -i ${INFILE}_d${DELTA_DIAGONALE}e${DELTA_PEAKS}i${MIN_INTENSITY} -o ${OUTFILE} -d ${DELTA_DIAGONALE} -f ${FC} -m ${MIN_READS_BIN} -r ${RESOLUTION}
perl ${TWEED_DIR}/HiChip_CTCF_coverage_TAD_with_peaks_filter_20180612.pl -i ${OUTDIR}/putative_tads -o ${OUTFILE} -d ${DELTA_DIAGONALE} -e ${DELTA_REF} -m ${MIN_READS_BIN} -r ${RESOLUTION}
perl ${TWEED_DIR}/HiChip_CTCF_coverage_TAD_merge_prediction_20190617.pl -i ${OUTFILE_DERIV} -j  ${OUTFILE_ENRICHED} -o ${OUTFILE_MERGED} -r ${RESOLUTION}
"	
			echo -e "${COMMAND}\n${PBS_OUT}/job_${JOBNAME}\n"		
			${COMMAND}
			#JOB=`echo -e "${COMMAND}\n" | qsub -e ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.err -m ae -o ${PBS_OUT}/job_${JOBNAME}.${PBS_JOBID}.out -N ${JOBNAME}   -q batch -l nodes=1:ppn=${NBPROC},mem=${MEMORY},walltime=${WALLTIME}`
			JOB=$(echo -e "${COMMAND}\n" | sbatch \
        --error=${PBS_OUT}/job_${JOBNAME}.%j.err \
        --output=${PBS_OUT}/job_${JOBNAME}.%j.out \
        --job-name=${JOBNAME} \
        --nodes=1 \
        --ntasks-per-node=${NBPROC} \
        --mem=${MEMORY} \
        --time=${WALLTIME})
			# exit 0
		fi

else
	echo "no step ${STEP} (-s option)"
	exit 0
fi
	