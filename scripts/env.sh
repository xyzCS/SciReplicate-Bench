#!/bin/bash
set -e  # Exit on any error
ENV_ROOT_DIR=$1
ENV_BENCHMARK_DIR=$2
# ENV_BENCHMARK_DIR="${ENVS_PATH}/Benchmark"
cd "${ENV_ROOT_DIR}"

# Check if envs_sci.zip exists
# if [ ! -f "envs_sci.zip" ]; then
#     echo "Error: envs_sci.zip not found"
#     exit 1
# fi

# unzip envs_sci.zip
conda init

# Source conda to make it available in this script
source $(conda info --base)/etc/profile.d/conda.sh

# Check if envs_sci directory exists, create if not
if [ ! -d "envs_sci" ]; then
    echo "Creating envs_sci directory..."
    mkdir -p envs_sci
fi

cd envs_sci
ENVS_PATH="${ENV_ROOT_DIR}/envs_sci"

echo "=========================================="
echo "Starting conda environments setup..."
echo "Root directory: ${ENV_ROOT_DIR}"
echo "Environments path: ${ENVS_PATH}"
echo "Benchmark directory: ${ENV_BENCHMARK_DIR}"
echo "=========================================="

# # 0-ColdFusion
# echo "Processing 0-ColdFusion..."
# if [ ! -f "0-ColdFusion.tar.gz" ]; then
#     echo "Error: 0-ColdFusion.tar.gz not found"
#     exit 1
# fi
# mkdir ColdFusion
# tar -xzf 0-ColdFusion.tar.gz -C ColdFusion
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="ColdFusion"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ ColdFusion environment setup complete"

# # 1-order
# echo "Processing 1-order..."
# if [ ! -f "1-order.tar.gz" ]; then
#     echo "Error: 1-order.tar.gz not found"
#     exit 1
# fi

# mkdir order
# tar -xzf 1-order.tar.gz -C order
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="order"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# pip install deepdiff
# conda deactivate
# echo "✓ order environment setup complete"

# # 2-gac_env
# echo "Processing 2-gac_env..."
# if [ ! -f "2-gac_env.tar.gz" ]; then
#     echo "Error: 2-gac_env.tar.gz not found"
#     exit 1
# fi

# mkdir gac_env
# tar -xzf 2-gac_env.tar.gz -C gac_env
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="gac_env"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ gac_env environment setup complete"

# # 3-tcom_env.tar.gz
# echo "Processing 3-tcom_env..."
# if [ ! -f "3-tcom_env.tar.gz" ]; then
#     echo "Error: 3-tcom_env.tar.gz not found"
#     exit 1
# fi

# mkdir tcom_env
# tar -xzf 3-tcom_env.tar.gz -C tcom_env
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="tcom_env"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ tcom_env environment setup complete"

# # 4-llm-planning 
# echo "Processing 4-llm-planning..."
# if [ ! -f "4-llm-planning.tar.gz" ]; then
#     echo "Error: 4-llm-planning.tar.gz not found"
#     exit 1
# fi

# mkdir llm-planning
# tar -xzf 4-llm-planning.tar.gz -C llm-planning
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="llm-planning"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"
# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ llm-planning environment setup complete"

# # 5-SRank
# echo "Processing 5-SRank..."
# if [ ! -f "5-SRank.tar.gz" ]; then
#     echo "Error: 5-SRank.tar.gz not found"
#     exit 1
# fi

# mkdir SRank
# tar -xzf 5-SRank.tar.gz -C SRank
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="SRank"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"
# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ 5-SRank environment setup complete"

# # 6-DEEIA
# echo "Processing 6-DEEIA..."
# if [ ! -f "6-DEEIA.tar.gz" ]; then
#     echo "Error: 6-DEEIA.tar.gz not found"
#     exit 1
# fi

# mkdir DEEIA
# tar -xzf 6-DEEIA.tar.gz -C DEEIA
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="DEEIA"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"
# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ 6-DEEIA environment setup complete"

# # 7-UniEA
# echo "Processing 7-UniEA..."
# if [ ! -f "7-UniEA.tar.gz" ]; then
#     echo "Error: 7-UniEA.tar.gz not found"
#     exit 1
# fi

# mkdir UniEA
# tar -xzf 7-UniEA.tar.gz -C UniEA
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="UniEA"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ 7-UniEA environment setup complete"

# # 8-CD
# echo "Processing 8-CD..."
# if [ ! -f "8-CD.tar.gz" ]; then
#     echo "Error: 8-CD.tar.gz not found"
#     exit 1
# fi

# mkdir CD
# tar -xzf 8-CD.tar.gz -C CD
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="CD"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ 8-CD environment setup complete"

# # 9-TransMI
# echo "Processing 9-TransMI..."
# if [ ! -f "9-TransMI.tar.gz" ]; then
#     echo "Error: 9-TransMI.tar.gz not found"
#     exit 1
# fi

# mkdir TransMI
# tar -xzf 9-TransMI.tar.gz -C TransMI
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="TransMI"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# conda deactivate
# echo "✓ 9-TransMI environment setup complete"

# # 10-rlkd
# echo "Processing 10-rlkd..."
# if [ ! -f "10-rlkd.tar.gz" ]; then
#     echo "Error: 10-rlkd.tar.gz not found"
#     exit 1
# fi

# mkdir rlkd
# tar -xzf 10-rlkd.tar.gz -C rlkd
# conda deactivate 2>/dev/null || true  # Don't fail if no env is active
# ENV_NAME="rlkd"
# ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# # Activate environment by path
# conda activate "${ENV_PATH}"
# conda-unpack
# pip uninstall torchsort
# cd "${ENV_BENCHMARK_DIR}"
# cd 10-RLKD-main
# cd torchsort-main
# pip install -e .
# conda deactivate
# cd "${ENVS_PATH}"
# echo "✓ 10-rlkd environment setup complete"

# 11-Averitec
echo "Processing 11-Averitec..."
if [ ! -f "11-Averitec.tar.gz" ]; then
    echo "Error: 11-Averitec.tar.gz not found"
    exit 1
fi

mkdir Averitec
tar -xzf 11-Averitec.tar.gz -C Averitec
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="Averitec"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
cd "${ENV_BENCHMARK_DIR}"
cd 11-AVeriTeC-DCE-main
cd AVeriTeC-DCE-main
pip install en_core_web_lg-3.8.0-py3-none-any.whl
conda deactivate
cd "${ENVS_PATH}"
echo '✓ 11-Averitec environment setup complete (with additional packages)'

# 12-IRCAN
echo "Processing 12-IRCAN..."
if [ ! -f "12-IRCAN.tar.gz" ]; then
    echo "Error: 12-IRCAN.tar.gz not found"
    exit 1
fi

mkdir IRCAN
tar -xzf 12-IRCAN.tar.gz -C IRCAN
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="IRCAN"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 12-IRCAN environment setup complete"

# 13-RouterDC
echo "Processing 13-RouterDC..."
if [ ! -f "13-RouterDC.tar.gz" ]; then
    echo "Error: 13-RouterDC.tar.gz not found"
    exit 1
fi

mkdir RouterDC
tar -xzf 13-RouterDC.tar.gz -C RouterDC
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="RouterDC"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 13-RouterDC environment setup complete"

# 14-alto
echo "Processing 14-alto..."
if [ ! -f "14-alto.tar.gz" ]; then
    echo "Error: 14-alto.tar.gz not found"
    exit 1
fi

mkdir alto
tar -xzf 14-alto.tar.gz -C alto
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="alto"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 14-alto environment setup complete"

# 15-raptor
echo "Processing 15-raptor..."
if [ ! -f "15-raptor.tar.gz" ]; then
    echo "Error: 15-raptor.tar.gz not found"
    exit 1
fi

mkdir raptor
tar -xzf 15-raptor.tar.gz -C raptor
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="raptor"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 15-raptor environment setup complete"

# 16-ken
echo "Processing 16-ken..."
if [ ! -f "16-ken.tar.gz" ]; then
    echo "Error: 16-ken.tar.gz not found"
    exit 1
fi

mkdir ken
tar -xzf 16-ken.tar.gz -C ken
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="ken"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 16-ken environment setup complete"

# 17-acs
echo "Processing 17-acs..."
if [ ! -f "17-acs.tar.gz" ]; then
    echo "Error: 17-acs.tar.gz not found"
    exit 1
fi

mkdir acs
tar -xzf 17-acs.tar.gz -C acs
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="acs"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
cd "${ENV_BENCHMARK_DIR}"
cd 17-Adaptive-Contrastive-Search-main
cd Adaptive-Contrastive-Search-main
cd transformers
pip install -e .
conda deactivate
cd "${ENVS_PATH}"
echo '✓ 17-acs environment setup complete (with transformers installation)'

# 18-minicheck
echo "Processing 18-minicheck..."
if [ ! -f "18-minicheck.tar.gz" ]; then
    echo "Error: 18-minicheck.tar.gz not found"
    exit 1
fi

mkdir minicheck
tar -xzf 18-minicheck.tar.gz -C minicheck
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="minicheck"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 18-minicheck environment setup complete"

# 19-nnn
echo "Processing 19-nnn..."
if [ ! -f "19-nnn.tar.gz" ]; then
    echo "Error: 19-nnn.tar.gz not found"
    exit 1
fi

mkdir nnn
tar -xzf 19-nnn.tar.gz -C nnn
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="nnn"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 19-nnn environment setup complete"

# 20-neuron
echo "Processing 20-neuron..."
if [ ! -f "20-neuron.tar.gz" ]; then
    echo "Error: 20-neuron.tar.gz not found"
    exit 1
fi

mkdir neuron
tar -xzf 20-neuron.tar.gz -C neuron
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="neuron"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 20-neuron environment setup complete"

# 21-ratescore
echo "Processing 21-ratescore..."
if [ ! -f "21-ratescore.tar.gz" ]; then
    echo "Error: 21-ratescore.tar.gz not found"
    exit 1
fi

mkdir ratescore
tar -xzf 21-ratescore.tar.gz -C ratescore
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="ratescore"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
pip uninstall numpy thinc spacy medspacy PyRuSH -y
pip install numpy
pip install thinc spacy medspacy PyRuSH
conda deactivate
echo "✓ 21-ratescore environment setup complete"

# 22-green_score
echo "Processing 22-green_score..."
if [ ! -f "22-green_score.tar.gz" ]; then
    echo "Error: 22-green_score.tar.gz not found"
    exit 1
fi

mkdir green_score
tar -xzf 22-green_score.tar.gz -C green_score
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="green_score"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 22-green_score environment setup complete"

# 23-sneuron
echo "Processing 23-sneuron..."
if [ ! -f "23-sneuron.tar.gz" ]; then
    echo "Error: 23-sneuron.tar.gz not found"
    exit 1
fi

mkdir sneuron
tar -xzf 23-sneuron.tar.gz -C sneuron
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="sneuron"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
pip uninstall vllm -y
pip install vllm==0.6.3.post1
conda deactivate
echo "✓ 23-sneuron environment setup complete"

# 25-rpo
echo "Processing 25-rpo..."
if [ ! -f "25-rpo.tar.gz" ]; then
    echo "Error: 25-rpo.tar.gz not found"
    exit 1
fi

mkdir rpo
tar -xzf 25-rpo.tar.gz -C rpo
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="rpo"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 25-rpo environment setup complete"

# 26-recipe
echo "Processing 26-recipe..."
if [ ! -f "26-recipe.tar.gz" ]; then
    echo "Error: 26-recipe.tar.gz not found"
    exit 1
fi

mkdir recipe
tar -xzf 26-recipe.tar.gz -C recipe
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="recipe"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 26-recipe environment setup complete"

# 27-neuromax
echo "Processing 27-neuromax..."
if [ ! -f "27-neuromax.tar.gz" ]; then
    echo "Error: 27-neuromax.tar.gz not found"
    exit 1
fi

mkdir neuromax
tar -xzf 27-neuromax.tar.gz -C neuromax
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="neuromax"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 27-neuromax environment setup complete"

# 28-bridging
echo "Processing 28-bridging..."
if [ ! -f "28-bridging.tar.gz" ]; then
    echo "Error: 28-bridging.tar.gz not found"
    exit 1
fi

mkdir bridging
tar -xzf 28-bridging.tar.gz -C bridging
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="bridging"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 28-bridging environment setup complete"

# 29-degcg
echo "Processing 29-degcg..."
if [ ! -f "29-degcg.tar.gz" ]; then
    echo "Error: 29-degcg.tar.gz not found"
    exit 1
fi

mkdir degcg
tar -xzf 29-degcg.tar.gz -C degcg
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="degcg"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 29-degcg environment setup complete"

# 30-SafeDecoding
echo "Processing 30-SafeDecoding..."
if [ ! -f "30-SafeDecoding.tar.gz" ]; then
    echo "Error: 30-SafeDecoding.tar.gz not found"
    exit 1
fi

mkdir SafeDecoding
tar -xzf 30-SafeDecoding.tar.gz -C SafeDecoding
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="SafeDecoding"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 30-SafeDecoding environment setup complete"

# 31-MaskLID
echo "Processing 31-MaskLID..."
if [ ! -f "31-MaskLID.tar.gz" ]; then
    echo "Error: 31-MaskLID.tar.gz not found"
    exit 1
fi

mkdir MaskLID
tar -xzf 31-MaskLID.tar.gz -C MaskLID
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="MaskLID"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 31-MaskLID environment setup complete"

# 32-clear
echo "Processing 32-clear..."
if [ ! -f "32-clear.tar.gz" ]; then
    echo "Error: 32-clear.tar.gz not found"
    exit 1
fi

mkdir clear
tar -xzf 32-clear.tar.gz -C clear
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="clear"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 32-clear environment setup complete"

# 33-distill
echo "Processing 33-distill..."
if [ ! -f "33-distill.tar.gz" ]; then
    echo "Error: 33-distill.tar.gz not found"
    exit 1
fi

mkdir distill
tar -xzf 33-distill.tar.gz -C distill
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="distill"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 33-distill environment setup complete"

# 34-gliner
echo "Processing 34-gliner..."
if [ ! -f "34-gliner.tar.gz" ]; then
    echo "Error: 34-gliner.tar.gz not found"
    exit 1
fi

mkdir gliner
tar -xzf 34-gliner.tar.gz -C gliner
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="gliner"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 34-gliner environment setup complete"

# 35-beam
echo "Processing 35-beam..."
if [ ! -f "35-beam.tar.gz" ]; then
    echo "Error: 35-beam.tar.gz not found"
    exit 1
fi

mkdir beam
tar -xzf 35-beam.tar.gz -C beam
conda deactivate 2>/dev/null || true  # Don't fail if no env is active
ENV_NAME="beam"
ENV_PATH="${ENVS_PATH}/${ENV_NAME}"

# Activate environment by path
conda activate "${ENV_PATH}"
conda-unpack
conda deactivate
echo "✓ 35-beam environment setup complete"

echo "=========================================="
echo "🎉 All 36 conda environments have been successfully extracted and unpacked!"
echo "Total environments processed: 36"
echo "=========================================="