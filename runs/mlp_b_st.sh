#!/bin/sh

# [--pcap] [--reextract] <dataset_path> <dataset> <attack> <num_classes> [gpu_id|cpu] [DATE_TIME]

CALL_DIR=$(pwd)

cd ..

INPUT="csv"
REEXTRACT=0

while true; do
	case "$1" in
		--pcap)			INPUT="pcap"; shift ;;
		--reextract)	REEXTRACT=1; shift ;;
		*)				break ;;
	esac
done

if [ -z "$6" ]
then
	DATE_TIME=$(date +%Y-%m-%d-%H-%M-%S)
else
	DATE_TIME=$6
fi

MODEL="mlp_b_st"
DATASET_PATH=${1%/}
DATASET=$2

case "${DATASET_PATH}" in
	/*)	;;
	*)	DATASET_PATH="${CALL_DIR}"/"${DATASET_PATH}" ;;
esac
ATTACK=$3
NUM_CLASSES=$4
REQUESTED_DEVICE=${5:-0}

if [ "${REQUESTED_DEVICE}" = "cpu" ]; then
	DEVICE=cpu
elif python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
	DEVICE=${REQUESTED_DEVICE}
else
	echo "CUDA is not available in this environment." >&2
	exit 1
fi

RUN_DIR=eval/"${MODEL}"/"${DATASET}"/"${ATTACK}"/"${DATE_TIME}"
EXTRACTOR=models/"${MODEL}"/pcap_feats.py

# Confidence threshold on the logit margin (highest minus second logit).
CONF_TAU=0.9

if [ "${INPUT}" = "pcap" ]; then
	TRAIN_DATA="${DATASET_PATH}"/"${ATTACK}"-train.pcap
	TEST_DATA="${DATASET_PATH}"/"${ATTACK}"-test.pcap
else
	TRAIN_DATA="${DATASET_PATH}"/"${ATTACK}"-train-"${MODEL}"-feats.csv
	TEST_DATA="${DATASET_PATH}"/"${ATTACK}"-test-"${MODEL}"-feats.csv
fi

mkdir -p "${RUN_DIR}"

echo "---------- ${MODEL} (device=${DEVICE}, input=${INPUT}) ----------"

echo "0. ${MODEL}: feature extraction settings$([ "${INPUT}" = "csv" ] && echo " and stored feature CSVs")."

# Records the extraction settings used for this run.
SETTINGS=$(python3 -c "import sys; sys.path.insert(0, 'models/${MODEL}'); \
import pcap_feats as P; \
print(f'flow_table_size={P.DEFAULT_FLOW_TABLE_SIZE} time_shift={P.DEFAULT_TIME_SHIFT} ' \
	  f'idle_timeout={P.DEFAULT_IDLE_TIMEOUT}')" 2>&1)
echo "${SETTINGS}" | tee "${RUN_DIR}"/"${MODEL}"_feats.log

if [ "${INPUT}" = "csv" ]; then
	for SPLIT in train test; do
		CSV="${DATASET_PATH}"/"${ATTACK}"-"${SPLIT}"-"${MODEL}"-feats.csv

		if [ "${REEXTRACT}" = 0 ] && [ -f "${CSV}" ]; then
			echo "Reusing ${CSV}" | tee -a "${RUN_DIR}"/"${MODEL}"_feats.log
			continue
		fi

		echo "Extracting ${CSV}" | tee -a "${RUN_DIR}"/"${MODEL}"_feats.log

		# Extract to a tmp file first.
		( python3 "${EXTRACTOR}"											\
			"${DATASET_PATH}"/"${ATTACK}"-"${SPLIT}".pcap					\
			"${CSV}".tmp													\
			--labels "${DATASET_PATH}"/"${ATTACK}"-"${SPLIT}"-labels.csv;	\
		  echo $? > "${RUN_DIR}"/.extract_status )							\
			2>&1 | tee -a "${RUN_DIR}"/"${MODEL}"_feats.log

		if [ "$(cat "${RUN_DIR}"/.extract_status)" != 0 ]; then
			rm -f "${CSV}".tmp "${RUN_DIR}"/.extract_status
			echo "Feature extraction failed for ${SPLIT}." >&2
			exit 1
		fi

		rm -f "${RUN_DIR}"/.extract_status
		mv "${CSV}".tmp "${CSV}"
	done
fi

echo "1. ${MODEL}: model training."

python3 -m models."${MODEL}"."${MODEL}"_train		\
	--cur_run "$DATE_TIME"							\
	--dataset "${ATTACK}"							\
	--cur_model "${MODEL}"							\
	--num_classes "${NUM_CLASSES}"					\
	--savepth eval/"${MODEL}"/"${DATASET}"			\
	--trainpth "${TRAIN_DATA}"						\
	--testpth "${TEST_DATA}"						\
	--dlist 121										\
	--device "${DEVICE}"							\
	--modelnum 1									\
	2>&1 | tee "${RUN_DIR}"/"${MODEL}"_train.log

echo "2. ${MODEL}: conversion to LUT."

python3 -m models."${MODEL}"."${MODEL}"_lut_convert		\
	--dataset "${ATTACK}"								\
	--num_classes "${NUM_CLASSES}"						\
	--savepth "${RUN_DIR}"/"${MODEL}"_lut_convert.pt	\
	--testpth "${TEST_DATA}"							\
	--dlist 121											\
	--ptpth "${RUN_DIR}"/weight/"${MODEL}"_4_best.pt	\
	2>&1 | tee "${RUN_DIR}"/"${MODEL}"_lut_convert.log

echo "3. ${MODEL}: conversion from .pt to .pkl (MAT format), checked against the LUT model."

python3 -m models."${MODEL}"."${MODEL}"_pt_to_pkl						\
	--model_path "${RUN_DIR}"/"${MODEL}"_lut_convert.pt					\
	--cur_model "${MODEL}"												\
	--test_data_path "${TEST_DATA}"										\
	--num_classes "${NUM_CLASSES}"										\
	--output_pkl_path "${RUN_DIR}"/"${MODEL}"_lut_convert_dp.pkl		\
	--metadata_pkl_path "${RUN_DIR}"/"${MODEL}"_lut_convert_dp_meta.pkl	\
	--check																\
	--conf_tau "${CONF_TAU}"											\
	--packets_csv														\
	--dataset "${DATASET}"												\
	--attack "${ATTACK}"												\
	--date_time "${DATE_TIME}"											\
	2>&1 | tee "${RUN_DIR}"/"${MODEL}"_pt_to_pkl.log
