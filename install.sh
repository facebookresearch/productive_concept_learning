# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Installation for setting up productive concept learning project.
python3 -m venv release
source release/bin/activate

pip3 install torch torchvision torchaudio
pip3 install fairseq==v0.9.0

pip install OmegaConf==1.4.1
pip install hydra-core==0.11
pip install tensorboardX
pip install tensorboard
pip install soundfile
pip install tqdm
pip install scipy
pip install matplotlib
pip install scikit-learn==0.22.0
pip install frozendict
pip install pandas
pip install submitit