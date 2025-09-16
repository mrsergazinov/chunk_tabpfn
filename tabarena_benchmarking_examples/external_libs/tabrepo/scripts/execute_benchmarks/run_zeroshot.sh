#!/bin/bash

# Install code remotely
set -e

echo "Working directory: $PWD"

# Verify the runbenchmark.py file exists in the expected location
FILE_RUNBENCHMARK="../automlbenchmark/runbenchmark.py"
if [ -f "$FILE_RUNBENCHMARK" ]; then
    echo "$FILE_RUNBENCHMARK exists."
else
    echo "ERROR: $FILE_RUNBENCHMARK does not exist."
    exit 1
fi

# Copy the custom configs to the current directory so we can specify them in the runbenchmark.py arguments
cp -r ../automlbenchmark/custom_configs/ ./

COMMAND_STR="python ../automlbenchmark/runbenchmark.py"
BENCHMARK_STR="ag_244"
CONFIGS_CONSTRAINT_STR="60h8c"
FOLD_STR="-f 0 1 2"
AWS_STR="-m aws -p 3000"
CUSTOM_CONFIGS_STR="-u custom_configs"

# Run the full TabRepo experiments to generate the pre-computed results for all 1310 configs
# This will launch 92,232 m6i.2xlarge EC2 instances on AWS, each running for ~1 hour.
# This script will only work as-is out of the box if you have an AWS account with extremely large instance limit permissions.
# Otherwise, you will need to edit the max parallelization value.
# Wait 6500 seconds between each command
#  6500 seconds due to 8 seconds between instance startup, and needing to start 244*3 instances = 5856 seconds, plus some overhead
#  126 commands to execute, for a total of 6500*126 seconds = 819000 seconds = 228 hours = 10 days
SLEEP_TIME=6500

BASELINES=(
  "AutoGluon_bq:latest"
  "AutoGluon_hq:latest"
  "AutoGluon_mq:latest"
  "H2OAutoML:2023Q2"
  "autosklearn:2023Q2"
  "autosklearn2:2023Q2"
  "flaml:2023Q2"
  "GAMA_benchmark:2023Q2"
  "lightautoml:2023Q2"
  "mljarsupervised_benchmark:2023Q2"
  "constantpredictor:2023Q2"
  "RandomForest:2023Q2"
  "TunedRandomForest:2023Q2"
)

BASELINES=("${BASELINES[@]/%/ $BENCHMARK_STR}")
BASELINES_1h8c=("${BASELINES[@]/%/ 1h8c}")
BASELINES_4h8c=("${BASELINES[@]/%/ 4h8c}")

CONFIGS=(
  "ZS_BAG_catboost_b0:zeroshot"
  "ZS_BAG_catboost_b1:zeroshot"
  "ZS_BAG_catboost_b2:zeroshot"
  "ZS_BAG_catboost_b3:zeroshot"
  "ZS_BAG_catboost_b4:zeroshot"
  "ZS_BAG_catboost_b5:zeroshot"
  "ZS_BAG_catboost_b6:zeroshot"
  "ZS_BAG_catboost_b7:zeroshot"
  "ZS_BAG_catboost_b8:zeroshot"
  "ZS_BAG_catboost_b9:zeroshot"
  "ZS_BAG_catboost_b10:zeroshot"
  "ZS_BAG_catboost_b11:zeroshot"
  "ZS_BAG_catboost_b12:zeroshot"
  "ZS_BAG_lightgbm_b0:zeroshot"
  "ZS_BAG_lightgbm_b1:zeroshot"
  "ZS_BAG_lightgbm_b2:zeroshot"
  "ZS_BAG_lightgbm_b3:zeroshot"
  "ZS_BAG_lightgbm_b4:zeroshot"
  "ZS_BAG_lightgbm_b5:zeroshot"
  "ZS_BAG_lightgbm_b6:zeroshot"
  "ZS_BAG_lightgbm_b7:zeroshot"
  "ZS_BAG_lightgbm_b8:zeroshot"
  "ZS_BAG_lightgbm_b9:zeroshot"
  "ZS_BAG_lightgbm_b10:zeroshot"
  "ZS_BAG_lightgbm_b11:zeroshot"
  "ZS_BAG_lightgbm_b12:zeroshot"
  "ZS_BAG_xgboost_b0:zeroshot"
  "ZS_BAG_xgboost_b1:zeroshot"
  "ZS_BAG_xgboost_b2:zeroshot"
  "ZS_BAG_xgboost_b3:zeroshot"
  "ZS_BAG_xgboost_b4:zeroshot"
  "ZS_BAG_xgboost_b5:zeroshot"
  "ZS_BAG_xgboost_b6:zeroshot"
  "ZS_BAG_xgboost_b7:zeroshot"
  "ZS_BAG_xgboost_b8:zeroshot"
  "ZS_BAG_xgboost_b9:zeroshot"
  "ZS_BAG_xgboost_b10:zeroshot"
  "ZS_BAG_xgboost_b11:zeroshot"
  "ZS_BAG_xgboost_b12:zeroshot"
  "ZS_BAG_fastai_b0:zeroshot"
  "ZS_BAG_fastai_b1:zeroshot"
  "ZS_BAG_fastai_b2:zeroshot"
  "ZS_BAG_fastai_b3:zeroshot"
  "ZS_BAG_fastai_b4:zeroshot"
  "ZS_BAG_fastai_b5:zeroshot"
  "ZS_BAG_fastai_b6:zeroshot"
  "ZS_BAG_fastai_b7:zeroshot"
  "ZS_BAG_fastai_b8:zeroshot"
  "ZS_BAG_fastai_b9:zeroshot"
  "ZS_BAG_fastai_b10:zeroshot"
  "ZS_BAG_fastai_b11:zeroshot"
  "ZS_BAG_fastai_b12:zeroshot"
  "ZS_BAG_nn_torch_b0:zeroshot"
  "ZS_BAG_nn_torch_b1:zeroshot"
  "ZS_BAG_nn_torch_b2:zeroshot"
  "ZS_BAG_nn_torch_b3:zeroshot"
  "ZS_BAG_nn_torch_b4:zeroshot"
  "ZS_BAG_nn_torch_b5:zeroshot"
  "ZS_BAG_nn_torch_b6:zeroshot"
  "ZS_BAG_nn_torch_b7:zeroshot"
  "ZS_BAG_nn_torch_b8:zeroshot"
  "ZS_BAG_nn_torch_b9:zeroshot"
  "ZS_BAG_nn_torch_b10:zeroshot"
  "ZS_BAG_nn_torch_b11:zeroshot"
  "ZS_BAG_nn_torch_b12:zeroshot"
  "ZS_BAG_rf_b0:zeroshot"
  "ZS_BAG_rf_b1:zeroshot"
  "ZS_BAG_rf_b2:zeroshot"
  "ZS_BAG_rf_b3:zeroshot"
  "ZS_BAG_rf_b4:zeroshot"
  "ZS_BAG_rf_b5:zeroshot"
  "ZS_BAG_rf_b6:zeroshot"
  "ZS_BAG_rf_b7:zeroshot"
  "ZS_BAG_rf_b8:zeroshot"
  "ZS_BAG_rf_b9:zeroshot"
  "ZS_BAG_rf_b10:zeroshot"
  "ZS_BAG_rf_b11:zeroshot"
  "ZS_BAG_rf_b12:zeroshot"
  "ZS_BAG_xt_b0:zeroshot"
  "ZS_BAG_xt_b1:zeroshot"
  "ZS_BAG_xt_b2:zeroshot"
  "ZS_BAG_xt_b3:zeroshot"
  "ZS_BAG_xt_b4:zeroshot"
  "ZS_BAG_xt_b5:zeroshot"
  "ZS_BAG_xt_b6:zeroshot"
  "ZS_BAG_xt_b7:zeroshot"
  "ZS_BAG_xt_b8:zeroshot"
  "ZS_BAG_xt_b9:zeroshot"
  "ZS_BAG_xt_b10:zeroshot"
  "ZS_BAG_xt_b11:zeroshot"
  "ZS_BAG_xt_b12:zeroshot"
  "ZS_BAG_lr_b0:zeroshot_v2"
  "ZS_BAG_lr_b1:zeroshot_v2"
  "ZS_BAG_lr_b2:zeroshot_v2"
  "ZS_BAG_lr_b3:zeroshot_v2"
  "ZS_BAG_knn_b0:zeroshot_v2"
  "ZS_BAG_knn_b1:zeroshot_v2"
  "ZS_BAG_knn_b2:zeroshot_v2"
  "ZS_BAG_knn_b3:zeroshot_v2"
  "ZS_BAG_tabpfn_b0:zeroshot_v2"
  "ZS_BAG_ftt_b0:zeroshot_ftt_gpu"
)

CONFIGS=("${CONFIGS[@]/%/ $BENCHMARK_STR}")
CONFIGS=("${CONFIGS[@]/%/ $CONFIGS_CONSTRAINT_STR}")

# Construct the terminal command for each run
COMMANDS=("${BASELINES_1h8c[@]}" "${BASELINES_4h8c[@]}" "${CONFIGS[@]}")
COMMANDS=("${COMMANDS[@]/#/$COMMAND_STR }")
COMMANDS=("${COMMANDS[@]/%/ $FOLD_STR}")
COMMANDS=("${COMMANDS[@]/%/ $AWS_STR}")
COMMANDS=("${COMMANDS[@]/%/ $CUSTOM_CONFIGS_STR}")
COMMANDS=("${COMMANDS[@]/%/ $RESUME_STR}")

i=0
NUM_COMMANDS=${#COMMANDS[@]}
for command in "${COMMANDS[@]}"; do
  echo "==================================="
  echo "Executing command: $command"
  ((i=i+1))
  LOG_FILE_NAME="log_$i.file"
  LOG_FILE_NAME=$(echo "$LOG_FILE_NAME" | tr '/\\' _)  # Remove / and \
  nohup $command > $LOG_FILE_NAME 2>&1 &
  echo "Command executed, logging to ${LOG_FILE_NAME}, sleeping for ${SLEEP_TIME}s... ($i of $NUM_COMMANDS commands executed)"

  sleep $SLEEP_TIME
done


echo "==================================="

echo "Done!"

exit 0
