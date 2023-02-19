export CUDA_VISIBLE_DEVICES="0"
pretrained_model_name="facebook/bart-large"
#pretrained_model_name="t5-base"
#pretrained_model_name="sentence-transformers/bert-base-nli-cls-token"
#pretrained_model_name="sentence-transformers/bert-base-nli-mean-tokens"
dataset_name="squad2"
data_dir="/mnt/d/MLData/data/unifiedqa"

python tasks/run_seq2seq_qa_train.py \
--dataset_name ${dataset_name} \
--data_dir ${data_dir} \
--pretrained_model_name ${pretrained_model_name} \
--gpus 1 \
--max_epochs 5 \
--lr 2e-4 \
--question_truncate 512 \
--answer_truncate 128 \
--batch_size 16 \
--num_workers 16 \
--precision 32 \
--project_name Seq2Seq_QA \
--default_root_dir ./experiments/logs \
--val_step_interval 5000 \
#--log_model \
