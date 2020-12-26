import torch

# SR : Segmentation Result
# GT : Ground Truth


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR >= threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR >= threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1).long()+(GT == 1).long()) == 2
    FN = ((SR == 0).long()+(GT == 1).long()) == 2

    SE = float(torch.sum(TP))/(float(torch.sum(TP.long()+FN.long())) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR >= threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).long()+(GT == 0).long()) == 2
    FP = ((SR == 1).long()+(GT == 0).long()) == 2

    SP = float(torch.sum(TN))/(float(torch.sum(TN.long()+FP.long())) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR >= threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).long()+(GT == 1).long()) == 2
    FP = ((SR == 1).long()+(GT == 0).long()) == 2

    PC = float(torch.sum(TP))/(float(torch.sum(TP.long()+FP.long())) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR >= threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.long()+GT.long()) == 2)
    Union = torch.sum((SR.long()+GT.long()) >= 1)

    JS = float(Inter)/(float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    with torch.no_grad():
        SR = SR >= threshold
        GT = GT == torch.max(GT)
        Inter = torch.sum((SR.long()+GT.long()) == 2, dim=-1)
        DC = 2.0 * Inter / (torch.sum(SR, dim=-1) + torch.sum(GT, dim=-1)).float()
        DC = DC.mean()    
    return DC.item()
