from torch import nn
from peft import get_peft_model, LoraConfig
from src.models.model.corigami_models import ConvTransModelSmall
from src.models.model.borzorogami_models import BorzoiOrogami, BorzoiOrogamiCTCF
from src.models.model.enformorogami_models import EnformerOrogamiDeep


def replace_bn_with_groupnorm(model):
    """
    Recursively replace all BatchNorm layers with GroupNorm(1, C)
    """
    for name, module in model.named_children():
        if "borzoi" in name:
            print(f"[DEBUG] Not changing this one: {name}")
            continue
        if isinstance(module, nn.BatchNorm1d):
            C = module.num_features
            gn = nn.GroupNorm(num_groups=1, num_channels=C)
            setattr(model, name, gn)
            print(f"[DEBUG] Set BN1d to GN1d: {name}")
        elif isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            gn = nn.GroupNorm(num_groups=1, num_channels=C)
            setattr(model, name, gn)
            print(f"[DEBUG] Set BN2d to GN2d: {name}")
        else:
            replace_bn_with_groupnorm(module)


def get_learnable_params(model, weight_decay=1e-5):
    # Make LoRA LR = head LR = 5e-4
    adapter_lr = 5e-4
    no_decay = []
    high_lr  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Biases, LayerNorm/LayerScale weights (i.e. 1D tensors) → no weight decay
        if len(param.shape) == 1:
            no_decay.append(param)
        else:
            # Everything else (including "lora" modules) → high_lr group
            high_lr.append(param)

    return [
        {'params': high_lr,   'weight_decay': 1e-5, 'lr': adapter_lr},
        {'params': no_decay,  'weight_decay': 0, 'lr': adapter_lr},
    ]

def set_lora(model):
    full_lora_modules = r"^borzoi\.(?!separable\d+).*conv_layer|^borzoi\..*to_q|^borzoi\..*to_v|^borzoi\.transformer\.\d+\.1\.fn\.1|^borzoi\.transformer\.\d+\.1\.fn\.4"
    transformer_modules = r"^borzoi\..*to_q|^borzoi\..*to_k|^borzoi\..*to_out|^borzoi\..*to_v|^borzoi\.transformer\.\d+\.1\.fn\.1|^borzoi\.transformer\.\d+\.1\.fn\.4"
    more_layer_modules = r"^borzoi\..*to_q|^borzoi\..*to_k|^borzoi\..*to_out|^borzoi\..*to_v|^borzoi\.transformer\.\d+\.1\.fn\.1|^borzoi\.transformer\.\d+\.1\.fn\.4|^borzoi\.unet1\..*conv_layer|^borzoi\.horizontal_conv\d+\..*conv_layer|^borzoi\.final_joined_convs\..*conv_layer|^borzoi\.upsampling_unet\d+\..*conv_layer"
    also_conv_layer_modules = r"^borzoi\..*to_q|^borzoi\..*to_k|^borzoi\..*to_out|^borzoi\..*to_v|^borzoi\.transformer\.\d+\.1\.fn\.1|^borzoi\.transformer\.\d+\.1\.fn\.4|^borzoi\.unet1\..*conv_layer|^borzoi\.horizontal_conv\d+\..*conv_layer|^borzoi\.final_joined_convs\..*conv_layer|^borzoi\.upsampling_unet\d+\..*conv_layer|^borzoi\.res_tower\.[68]\.conv_layer"

    only_linear = r"^borzoi\..*to_[qkv]|^borzoi\..*to_out|^borzoi\.transformer\.\d+\.1\.fn\.[14]|^borzoi\.horizontal_conv\d+\..*conv_layer|^borzoi\.final_joined_convs\..*conv_layer|^borzoi\.upsampling_unet\d+\..*conv_layer|^borzoi\.separable\d+\.conv_layer\.1"

    transformer_modules2 = r"^borzoi\..*to_q|^borzoi\..*to_v|^borzoi\.transformer\.\d+\.1\.fn\.1|^borzoi\.transformer\.\d+\.1\.fn\.4"

    lora_config = LoraConfig(
        target_modules=full_lora_modules#more_layer_modules#full_lora_modules,
    )
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        print(f"Layer: {name} with grads {param.requires_grad}")
        if "borzoi" in name:  # TODO: If I ever use another backbone, need to replace
            continue
        else:
            param.requires_grad = True # This sets the head params to training.
            print(f"[DEBUG] Setting grad to true for layer: {name}")
    return model


def get_model(args):
    if args.borzoi:
        model_type = "borzoi"
        if args.num_genom_feat > 0:
            model = BorzoiOrogamiCTCF(mid_hidden=128, local=args.local, model_type=model_type)
        else:
            model = BorzoiOrogami(mid_hidden=128, local=args.local, model_type=model_type)
    else:
        model = ConvTransModelSmall(mid_hidden=128, num_genomic_features=args.num_genom_feat)

    if args.use_groupnorm:
        replace_bn_with_groupnorm(model)  # by referemce

    if args.lora:
        model = set_lora(model)

    return model

