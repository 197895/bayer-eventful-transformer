cd /home/huangjingkai/workspace/eventful-transformer
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate ev
export PYTHONPATH="$PYTHONPATH:."
# python scripts/evaluate/vitdet_vid.py base_672
# python scripts/evaluate/vitdet_vid.py bayer_temporal_full_1024
python scripts/evaluate/vitdet_vid.py bayer_temporal_full_672