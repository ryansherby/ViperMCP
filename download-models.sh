#download-models.sh
#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PRETRAINED_MODELS_PATH="$SCRIPT_DIR/viper/pretrained_models"

# ------------ Create directories for pretrained models ----------
for d in midas blip xvlm bert llama codex dino sam; do
  mkdir -p "$PRETRAINED_MODELS_PATH/$d"
done
mkdir -p "$PRETRAINED_MODELS_PATH/glip/checkpoints" "$PRETRAINED_MODELS_PATH/glip/configs"

# ---------- Depth-estimation (DPT-Hybrid MiDaS) ----------
huggingface-cli download Intel/dpt-hybrid-midas \
  --local-dir "$PRETRAINED_MODELS_PATH/midas" \
  --local-dir-use-symlinks False

# ---------- Simple VQA (BLIP-2 Flan T5 xl) ----------
# huggingface-cli download Salesforce/blip2-flan-t5-xl \
#   --local-dir "$PRETRAINED_MODELS_PATH/blip" \
#   --local-dir-use-symlinks False \
#   --include "*.safetensors" \

# ---------- BERT tokenizer ----------
huggingface-cli download google-bert/bert-base-uncased \
  --local-dir "$PRETRAINED_MODELS_PATH/bert" \
  --local-dir-use-symlinks False \
  --include "*.json" "*.txt" "*.safetensors"

# ---------- Similarity (XVLM) ----------
if [ ! -f "$PRETRAINED_MODELS_PATH/xvlm/retrieval_mscoco_checkpoint_9.pth" ]; then
  echo "Downloading XVLM model..."
  gdown "https://drive.google.com/uc?id=1bv6_pZOsXW53EhlwU0ZgSk03uzFI61pN" \
  -O "$PRETRAINED_MODELS_PATH/xvlm/retrieval_mscoco_checkpoint_9.pth"
else
  echo "XVLM model already exists, skipping download."
fi


## ---------- Object detection (GLIP) ---------- 
#wget -nc -P "$PRETRAINED_MODELS_PATH/glip/checkpoints" \
#  https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
#wget -nc -P "$PRETRAINED_MODELS_PATH/glip/configs" \
#  https://raw.githubusercontent.com/microsoft/GLIP/main/configs/pretrain/glip_Swin_L.yaml

# ---------- (Optional) Llama-2 7B ----------
#   ~13 GB download; uncomment if/when you really need it
# huggingface-cli download NousResearch/Llama-2-7b-hf \
#   --local-dir "$PRETRAINED_MODELS_PATH/llama" \
#   --local-dir-use-symlinks False \
#  --exclude "*.bin" \

# ---------- Object Detection (Grounding DINO) ----------
huggingface-cli download IDEA-Research/grounding-dino-base \
  --local-dir "$PRETRAINED_MODELS_PATH/dino" \
  --local-dir-use-symlinks False \
  --exclude "*.bin" \

# ---------- Segment Anything (SAM) ----------
huggingface-cli download facebook/sam-vit-base \
  --local-dir "$PRETRAINED_MODELS_PATH/sam" \
  --local-dir-use-symlinks False \
  --exclude "*.bin" \

# ---------- Code Llama 7B Instruct ----------
# huggingface-cli download codellama/CodeLlama-7b-Instruct-hf \
#  --local-dir "$PRETRAINED_MODELS_PATH/codex" \
#  --local-dir-use-symlinks False \
#  --exclude "*.bin" \

