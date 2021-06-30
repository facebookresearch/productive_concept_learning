# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Meta-learning dataset properties
#
# Provides the paths relevant to the meta-dataset generation code
OUTPUT_DATASET_PATH="/checkpoint/ramav/adhoc_concept_data/"
RAW_IMAGE_DATA_PATH="adhoc_images_slurm_v0.2"

# 
DATASET_PROPERTIES="clevr_typed_fol_properties"
GRAMMAR_TYPE="v2_typed_simple_fol"
MAX_RECURSION_DEPTH=6
NUM_HYPOTHESES=2000000
BAN_STRINGS_WITH_SAME_ARGS=1
MAX_SCENE_FILE_ID_FOR_EXEC=200
RAW_DATASET_NUM_IMAGES=5000000
META_DATASET_TRAIN_SIZE=500000

META_DATASET_NAME=$(cat <<EOF
${GRAMMAR_TYPE}_\
depth_${MAX_RECURSION_DEPTH}_trials_${NUM_HYPOTHESES}_\
ban_${BAN_STRINGS_WITH_SAME_ARGS}\
_max_scene_id_${MAX_SCENE_FILE_ID_FOR_EXEC}
EOF
)


JSON_EXEC_RES_FILES_PREFIX=$(cat <<EOF
${OUTPUT_DATASET_PATH}/${RAW_IMAGE_DATA_PATH}/\
hypotheses/result_${META_DATASET_NAME}
EOF
)

META_DATASET_PATH=$(cat <<EOF
${OUTPUT_DATASET_PATH}/${RAW_IMAGE_DATA_PATH}/hypotheses/${META_DATASET_NAME}
EOF
)