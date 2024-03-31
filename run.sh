sleep $1
pip install accelerate pandas transformers
accelerate launch train.py $2