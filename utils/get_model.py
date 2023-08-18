from models import inception_v1, mobilenet_v2, resNeXt50, vgg19, resnet18, resnet50


def get_model(str_):
    if str_ == "efficientnet_b0":
        pass

    elif str_ == "inception_v1":
        return inception_v1.GoogleNet()

    elif str_ == "mobilenet_v2":
        return mobilenet_v2.Mobilenet_V2()

    elif str_ == "resNeXt50":
        return resNeXt50.ResNeXt_50()

    elif str_ == "vgg19":
        return vgg19.VGG19()

    elif str_ == "resnet18":
        return resnet18.ResNet_18()

    elif str_ == "resnet50":
        return resnet50.ResNet_50()

    else:
        raise ValueError("Invalid model name")

    print(f"{str_} loaded successfully")
