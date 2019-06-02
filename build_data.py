import json
import numpy as np
from utils import read_properties,collect_BIO2id,collect_char2id,collect_data_set,collect_word2id,readFile,collect_n_char2id

config_file= read_properties('config/CoNLL04/bio_config')
filename_train = config_file.getProperty("filename_train")
filename_test = config_file.getProperty("filename_test")
filename_dev = config_file.getProperty("filename_dev")

filename_train_me = config_file.getProperty("filename_train_me")
filename_test_me = config_file.getProperty("filename_test_me")
filename_dev_me = config_file.getProperty("filename_dev_me")

filename_char2id = config_file.getProperty("filename_char2id")
filename_n_char2id = config_file.getProperty("filename_n_char2id")

filename_word2id = config_file.getProperty("filename_word2id")
filename_BIO2id = config_file.getProperty("filename_BIO2id")
filename_relation2id = config_file.getProperty("filename_relation2id")

train_data_me = collect_data_set(readFile(filename_train),filename_train_me)
dev_data_me = collect_data_set(readFile(filename_dev),filename_dev_me)
test_data_me = collect_data_set(readFile(filename_test),filename_test_me)

collect_char2id(train_data_me+dev_data_me+test_data_me,filename_char2id)
collect_n_char2id(train_data_me+dev_data_me+test_data_me,filename_n_char2id,3)
collect_word2id(train_data_me+dev_data_me+test_data_me,filename_word2id)
collect_BIO2id(train_data_me+dev_data_me+test_data_me,filename_BIO2id)
# collect_relations2id(train_data_me+dev_data_me+test_data_me,filename_relation2id)


