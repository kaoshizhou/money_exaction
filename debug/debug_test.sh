export PYTHONPATH=/data2/linyizhang/text_cls_pair:$PYTHONPATH
python3 at_template/client_test.py \
--dataset_path=/data2/linyizhang/text_cls_pair/datasets/scitail/ \
--model_path=/data2/linyizhang/text_cls_pair/temp/models/6FN3vFcqFcv3KdAB2jpAGu \
--used_model=BERT_autotables-2503-train-1 \
--runs_path=/data2/linyizhang/text_cls_pair/temp