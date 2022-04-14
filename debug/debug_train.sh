export PYTHONPATH=/home/jiawei/money_exaction:$PYTHONPATH
python3 at_template/client_train.py --dataset_path ./data/money_train.json --runs_path ./runs/ --device cuda