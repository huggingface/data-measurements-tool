#!/usr/bin/env bash


python3 run_data_measurements.py --dataset="hate_speech18" --config="default" --split="train" --label_field="label" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="hate_speech_offensive" --config="default" --split="train" --label_field="label" --feature="tweet" --overwrite_previous


python3 run_data_measurements.py --dataset="imdb" --config="plain_text" --split="train" --label_field="label" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="imdb" --config="plain_text" --split="unsupervised" --label_field="label" --feature="text" --overwrite_previous


python3 run_data_measurements.py --dataset="glue" --config="cola" --split="train" --label_field="label" --feature="sentence" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="cola" --split="validation" --label_field="label" --feature="sentence" --overwrite_previous

python3 run_data_measurements.py --dataset="glue" --config="mnli" --split="train" --label_field="label" --feature="hypothesis" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mnli" --split="train" --label_field="label" --feature="premise" --overwrite_previous

python3 run_data_measurements.py --dataset="glue" --config="mnli" --split="validation_matched" --label_field="label" --feature="premise" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mnli" --split="validation_matched" --label_field="label" --feature="hypothesis" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mnli" --split="validation_mismatched" --label_field="label" --feature="premise" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mnli" --split="validation_mismatched" --label_field="label" --feature="hypothesis" --overwrite_previous


python3 run_data_measurements.py --dataset="glue" --config="mrpc" --split="train" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mrpc" --split="train" --label_field="label" --feature="sentence2" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mrpc" --split="validation" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mrpc" --split="validation" --label_field="label" --feature="sentence2" --overwrite_previous


python3 run_data_measurements.py --dataset="glue" --config="rte" --split="train" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="rte" --split="train" --label_field="label" --feature="sentence2" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="rte" --split="validation" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="rte" --split="validation" --label_field="label" --feature="sentence2" --overwrite_previous


python3 run_data_measurements.py --dataset="glue" --config="stsb" --split="train" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="stsb" --split="train" --label_field="label" --feature="sentence2" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="stsb" --split="validation" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="stsb" --split="validation" --label_field="label" --feature="sentence2" --overwrite_previous

python3 run_data_measurements.py --dataset="glue" --config="wnli" --split="train" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="wnli" --split="train" --label_field="label" --feature="sentence2" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="wnli" --split="validation" --label_field="label" --feature="sentence1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="wnli" --split="validation" --label_field="label" --feature="sentence2" --overwrite_previous

python3 run_data_measurements.py --dataset="glue" --config="sst2" --split="train" --label_field="label" --feature="sentence" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="sst2" --split="validation" --label_field="label" --feature="sentence" --overwrite_previous


python3 run_data_measurements.py --dataset="glue" --config="qnli" --split="train" --label_field="label" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="qnli" --split="train" --label_field="label" --feature="sentence" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="qnli" --split="validation" --label_field="label" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="qnli" --split="validation" --label_field="label" --feature="sentence" --overwrite_previous


python3 run_data_measurements.py --dataset="glue" --config="qqp" --split="train" --label_field="label" --feature="question1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="qqp" --split="train" --label_field="label" --feature="question2" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="qqp" --split="validation" --label_field="label" --feature="question1" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="qqp" --split="validation" --label_field="label" --feature="question2" --overwrite_previous

python3 run_data_measurements.py --dataset="glue" --config="mnli_matched" --split="validation" --label_field="label" --feature="hypothesis" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mnli_matched" --split="validation" --label_field="label" --feature="premise" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mnli_mismatched" --split="validation" --label_field="label" --feature="hypothesis" --overwrite_previous
python3 run_data_measurements.py --dataset="glue" --config="mnli_mismatched" --split="validation" --label_field="label" --feature="premise" --overwrite_previous


python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-103-v1" --split="train" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-103-raw-v1" --split="train" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-2-v1" --split="train" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-2-raw-v1" --split="train" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-103-v1" --split="validation" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-103-raw-v1" --split="validation" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-2-v1" --split="validation" --feature="text" --overwrite_previous
python3 run_data_measurements.py --dataset="wikitext" --config="wikitext-2-raw-v1" --split="validation" --feature="text" --overwrite_previous


# Superglue wsc? wic? rte? record? multirc?

python3 run_data_measurements.py --dataset="super_glue" --config="boolq" --split="train" --label_field="label" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="boolq" --split="validation" --label_field="label" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="boolq" --split="train" --label_field="label" --feature="passage" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="boolq" --split="validation" --label_field="label" --feature="passage" --overwrite_previous

python3 run_data_measurements.py --dataset="super_glue" --config="cb" --split="train" --label_field="label" --feature="premise" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="cb" --split="validation" --label_field="label" --feature="premise" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="cb" --split="train" --label_field="label" --feature="hypothesis" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="cb" --split="validation" --label_field="label" --feature="hypothesis" --overwrite_previous


python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="train" --label_field="label" --feature="premise" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="validation" --label_field="label" --feature="premise" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="train" --label_field="label" --feature="choice1" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="validation" --label_field="label" --feature="choice1" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="train" --label_field="label" --feature="choice2" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="validation" --label_field="label" --feature="choice2" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="train" --label_field="label" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="super_glue" --config="copa" --split="validation" --label_field="label" --feature="question" --overwrite_previous

python3 run_data_measurements.py --dataset="squad" --config="plain_text" --split="train" --feature="context" --overwrite_previous
python3 run_data_measurements.py --dataset="squad" --config="plain_text" --split="train" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="squad" --config="plain_text" --split="train" --feature="title" --overwrite_previous
python3 run_data_measurements.py --dataset="squad" --config="plain_text" --split="validation" --feature="context" --overwrite_previous
python3 run_data_measurements.py --dataset="squad" --config="plain_text" --split="validation" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="squad" --config="plain_text" --split="validation" --feature="title" --overwrite_previous


python3 run_data_measurements.py --dataset="squad_v2" --config="squad_v2" --split="train" --feature="context" --overwrite_previous
python3 run_data_measurements.py --dataset="squad_v2" --config="squad_v2" --split="train" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="squad_v2" --config="squad_v2" --split="train" --feature="title" --overwrite_previous
python3 run_data_measurements.py --dataset="squad_v2" --config="squad_v2" --split="validation" --feature="context" --overwrite_previous
python3 run_data_measurements.py --dataset="squad_v2" --config="squad_v2" --split="validation" --feature="question" --overwrite_previous
python3 run_data_measurements.py --dataset="squad_v2" --config="squad_v2" --split="validation" --feature="title" --overwrite_previous
