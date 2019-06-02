import json
import utils
from utils import read_properties
# def NER_result_Evaluator(outputs,targets):
#     """
#     这里测评下ner的结果的f1
#     :return:
#     """
#
#     right,true,pred = 1e-10, 1e-10, 1e-10
#
#     for i in range(len(outputs)):
#         output = outputs[i]
#         target = targets[i]
#         output = output[:len(target)]
#         for j in range(len(output)):
#             if output[j] != 0:
#                 pred += 1
#                 if target[j] == output[j]:
#                     right += 1
#         for j in range(len(target)):
#             if target[j] != 0 :
#                 true+=1
#     R = right/pred
#     P = right/true
#     F = (2*P*R)/(P+R)
#     return P,R,F


def NER_result_Evaluator(outputs,targets):
    config_file = read_properties('config/CoNLL04/bio_config')
    filename_BIO2id = config_file.getProperty("filename_BIO2id")
    id2BIO, BIO2id = json.load(open(filename_BIO2id, encoding='utf-8'))
    right,true,pred = 1e-10, 1e-10, 1e-10
    for i in range(len(outputs)):
        output = outputs[i]
        target = targets[i]
        output = output[:len(target)]
        flag=  0
        output_pred = []
        target_pred = []
        for i in range(len(output)):
            bio = id2BIO[str(output[i])]
            if bio[0]=='B':
                output_pred.append(i)
                for j in range(i+1,len(output)):
                    bio = id2BIO[str(output[j])]
                    if bio[0] == 'I':
                        output_pred.append(i+j+1)
                    else:
                        break
                break
        for i in range(len(target)):
            bio = id2BIO[str(target[i])]
            if bio[0]=='B':
                target_pred.append(i)
                for j in range(i+1,len(target)):
                    bio = id2BIO[str(target[j])]
                    if bio[0] == 'I':
                        target_pred.append(i+j+1)
                    else:
                        break
                break

        if output_pred:
            pred+=1
        if target_pred:
            true+=1
        if output_pred == target_pred:
            right+=1

    R = right/pred
    P = right/true
    F = (2*P*R)/(P+R)
    return P,R,F
