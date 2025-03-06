import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch

coupling_block_dict = {
    "AllInOne":Fm.AllInOneBlock,
    "RQSpline":Fm.RationalQuadraticSpline
}

activation_dict = {
    "Tanh":torch.nn.Tanh,
    "ReLU":torch.nn.ReLU,
    "SiLU":torch.nn.SiLU,
    "ELU":torch.nn.ELU,
    "Sigmoid":torch.nn.Sigmoid,
    "Tanhshrink":torch.nn.Tanhshrink,
    "SoftSign":torch.nn.Softsign,
    "leakyReLU":torch.nn.LeakyReLU,
    "GeLU":torch.nn.GELU
    }

condition_specs_dict = {
        "ignore_beta":[None,None],
        "log_beta":[0,(1,)]
    }
