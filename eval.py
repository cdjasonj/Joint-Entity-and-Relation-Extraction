import json
import utils

def NER_result_Evaluator(outputs,targets):
    """
    这里测评下ner的结果的f1
    :return:
    """

    right,true,pred = 1e-10, 1e-10, 1e-10

    for i in range(len(outputs)):
        output = outputs[i]
        target = targets[i]
        output = output[:len(target)]
        for j in range(len(output)):
            if output[j] != 0:
                pred += 1
                if target[j] == output[j]:
                    right += 1
        for j in range(len(target)):
            if target[j] != 0 :
                true+=1
    R = right/pred
    P = right/true
    F = (2*P*R)/(P+R)
    return P,R,F