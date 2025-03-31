import torch.nn as nn
from torchvision import datasets, models, ops


def choose_model(model_name, class_num):
    if model_name == 'resnet18':
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18.fc = nn.Linear(resnet18.fc.in_features, class_num)
        model = resnet18
    if model_name == 'resnet50':
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet50.fc = nn.Linear(resnet50.fc.in_features, class_num)
        model = resnet50
    if model_name == 'resnet101':
        resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        resnet101.fc = nn.Linear(resnet101.fc.in_features, class_num)
        model = resnet101
    if model_name == 'efficientnetv2l':
        efficientV2L = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        efficientV2L.classifier[1] = nn.Linear(efficientV2L.classifier[1].in_features, class_num)
        model = efficientV2L
    if model_name == 'efficientnetv2m':
        efficientV2M = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        efficientV2M.classifier[1] = nn.Linear(efficientV2M.classifier[1].in_features, class_num)
        model = efficientV2M
    if model_name == 'efficientnetv2s':
        efficientV2S = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        efficientV2S.classifier[1] = nn.Linear(efficientV2S.classifier[1].in_features, class_num)
        model = efficientV2S
    if model_name == 'swinv2b':
        swimv2b = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
        swimv2b.head = nn.Linear(swimv2b.head.in_features, class_num)
        model = swimv2b
    if model_name == 'swinv2s':
        swimv2s = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)
        swimv2s.head = nn.Linear(swimv2s.head.in_features, class_num)
        model = swimv2s
    if model_name == 'swinv2t':
        swimv2t = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        swimv2t.head = nn.Linear(swimv2t.head.in_features, class_num)
        model = swimv2t

    return model


if __name__ == '__main__':
    model = choose_model('resnet50')
    print(model)