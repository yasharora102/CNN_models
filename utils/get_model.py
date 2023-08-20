from models import inception_v1, mobilenet_v2, resNeXt50, vgg19, resnet18, resnet50,efficientnet_b0


def get_model(str_, num_classes):
    if str_ == "efficientnet_b0":
        return efficientnet_b0.EfficientNet_b0(num_classes)

    elif str_ == "inception_v1":
        return inception_v1.GoogleNet(num_classes)

    elif str_ == "mobilenet_v2":
        return mobilenet_v2.Mobilenet_V2(num_classes)

    elif str_ == "resNeXt50":
        return resNeXt50.ResNeXt_50(num_classes)

    elif str_ == "vgg19":
        return vgg19.VGG19(num_classes)

    elif str_ == "resnet18":
        return resnet18.ResNet_18(num_classes)

    elif str_ == "resnet50":
        return resnet50.ResNet_50(num_classes)

    else:
        raise ValueError("Invalid model name")
