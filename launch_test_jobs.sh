# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Run jobs on TEST set
source release/bin/activate
source paths.sh

############################## Global parameters ###############################
METRIC="modelmap"
GREATER_BETTER="1"
JOB_TYPE="eval"
REPLICA_JOBS="0,1,2" # 3 jobs per run.
MODEL_OR_ORACLE="model" # Compute oracle metrics separately


################################### Model sweep ################################
JOB_NAME="sweep_models"
SWEEP_PATH="${RUN_DIR}/${JOB_NAME}"
python scripts/pick_best_checkpoints.py \
  --sweep_folder ${SWEEP_PATH} \
  --val_metric ${METRIC} \
  --greater_is_better ${GREATER_BETTER}

# Sweep that trains all the models explained in the paper.
for SPLIT_TYPE in "comp" "iid" "color_count" "color_location" "color_material" \
    "color" "shape" "color_boolean" "length_threshold_10"
do
    for MODALITY in "image" "json" "sound"
    do
        for POOLING in "gap" "concat" "rel_net" "trnsf"
        do
            for LANGUAGE_ALPHA in "0.0" "1.0"
            do 
                for NEGATIVE_TYPE in "alternate_hypotheses" "random"
                do
                    for REPLICA_JOBS in "0" "1" "2"
                    do
    				    EVAL_STR=""
                        if [ $JOB_TYPE = "eval" ];
                        then
                            EVAL_STR="eval_cfg.write_raw_metrics=True"
                        fi
                        CMD="python hydra_${JOB_TYPE}.py \
  	  					  splits='test & cross_split'\
                          eval_split_name=test\
                          eval_cfg.best_test_metric=${METRIC} \
    					  ${EVAL_STR}\
                          hydra.job.name=${JOB_NAME}\
                          mode=${JOB_TYPE} \
                          model_or_oracle_metrics=${MODEL_OR_ORACLE} \
                          modality=${MODALITY}\
                          pooling=${POOLING}\
                          data.split_type=${SPLIT_TYPE} \
                          data.negative_type=${NEGATIVE_TYPE}\
                          loss.params.alpha=${LANGUAGE_ALPHA} \
                          job_replica=${REPLICA_JOBS} &"
                        echo ${CMD}
                        eval ${CMD}
                    done
                done
            done
        done
    done
done


############################## Oracle Metrics ########################################
JOB_NAME="oracle" # Test oracle jobs
SWEEP_PATH="${RUN_DIR}/${JOB_NAME}"

MODEL_OR_ORACLE="oracle" # Compute oracle metrics separately
LANGUAGE_ALPHA="0.0"
REPLICA_JOBS="0"
MODALITY="json"
POOLING="gap"

NEGATIVE_TYPE="alternate_hypotheses,random"

# Jobs are launched with for loops to avoid MaxJobArrayLimit on SLURM.
for SPLIT_TYPE in "iid" "color_count" "color_location" "color_material"\
    "color" "shape" "color_boolean" "length_threshold_10" "comp"
do
    for NEGATIVE_TYPE in "alternate_hypotheses" "random"
    do
        CMD="python hydra_${JOB_TYPE}.py \
				  splits='test & cross_split'\
                  eval_split_name=test\
                  eval_cfg.best_test_metric=${METRIC} \
                  hydra.job.name=${JOB_NAME}\
                  mode=${JOB_TYPE} \
                  model_or_oracle_metrics=${MODEL_OR_ORACLE} \
                  modality=${MODALITY}\
                  pooling=${POOLING}\
                  data.split_type=${SPLIT_TYPE} \
                  data.negative_type=${NEGATIVE_TYPE}\
                  loss.params.alpha=${LANGUAGE_ALPHA} \
                  job_replica=${REPLICA_JOBS} &"
        echo ${CMD}
        eval ${CMD}
    done
done