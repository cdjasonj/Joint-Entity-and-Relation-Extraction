import json
import utils

def NER_result_Evaluator(outputs,targets):
    """
    这里测评下ner的结果的f1
    :return:
    """
    R, P, F = 1e-10, 1e-10, 1e-10
    right = 0
    true = 0
    pred = 0
    assert len(outputs) == len(targets)
    for i in range(len(outputs)):
        output = outputs[i]
        target = targets[i]
        for j in range(len(target)):
            if output != 0:
                pred += 1
            if target != 0:
                true+= 1
                if target == output:
                    right += 1
    P = pred/true
    R = right/true
    F = (2*P*R)/(P+R)
    return P,R,F