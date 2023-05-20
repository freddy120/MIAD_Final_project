cd ~/MIAD_Final_project/etl/
conda activate etl
python3 etl_first.py

cd ~/MIAD_Final_project/training/
conda activate api
python3 validation.py
python3 train_cpu.py

cd ~/MIAD_Final_project/predictions/
python3 predictions.py
