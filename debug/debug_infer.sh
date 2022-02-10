export PYTHONPATH=/data2/linyizhang/text_cls_pair:$PYTHONPATH
python3 at_template/client_infer.py \
--model_path=/data2/linyizhang/text_cls_pair/temp/models/6FN3vFcqFcv3KdAB2jpAGu \
--used_model=BERT_autotables-2503-train-1 \
--service_port=8080