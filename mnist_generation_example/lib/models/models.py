import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.models.model_utils as model_utils
from torchtyping import TensorType
import torch.autograd.profiler as profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.models.forward_model import (
    GaussianTargetRate,
)
# path to 
from lib.networks.dit import DiT
from lib.networks.uvit import UViT

def center_data(x, x_min_max):
  out = (x - x_min_max[0]) / (x_min_max[1] - x_min_max[0]) # [0, 1]
  return 2 * out - 1 

def log_minus_exp(a, b, eps=1e-6):
    """
    Compute log (exp(a) - exp(b)) for (b<a)
    From https://arxiv.org/pdf/2107.03006.pdf
    """
    return a + torch.log1p(-torch.exp(b - a) + eps)


def sample_logistic(net_out, B, C, D, S, fix_logistic, device):
    """
    net_out: Output of neural network with shape B, 2*C, H,W
    B: Batch Size
    C: Number of channel
    D: Dimension = C*H*W
    S: Number of States

    """

    mu = net_out[0].unsqueeze(-1)  # B, C, H, W, 1
    log_scale = net_out[1].unsqueeze(-1)  # B, C, H, W, 1

    # if self.padding:
    #    mu = mu[:, :, :-1, :-1, :]
    #    log_scale = log_scale[:, :, :-1, :-1, :]

    # The probability for a state is then the integral of this continuous distribution between
    # this state and the next when mapped onto the real line. To impart a residual inductive bias
    # on the output, the mean of the logistic distribution is taken to be tanh(xt + μ′) where xt
    # is the normalized input into the model and μ′ is mean outputted from the network.
    # The normalization operation takes the input in the range 0, . . . , 255 and maps it to [−1, 1].
    inv_scale = torch.exp(-(log_scale - 2))

    bin_width = 2.0 / S
    bin_centers = torch.linspace(
        start=-1.0 + bin_width / 2,
        end=1.0 - bin_width / 2,
        steps=S,
        device=device,
    ).view(1, 1, 1, 1, S)

    sig_in_left = (bin_centers - bin_width / 2 - mu) * inv_scale
    bin_left_logcdf = F.logsigmoid(sig_in_left)
    sig_in_right = (bin_centers + bin_width / 2 - mu) * inv_scale
    bin_right_logcdf = F.logsigmoid(sig_in_right)

    logits_1 = log_minus_exp(bin_right_logcdf, bin_left_logcdf)
    logits_2 = log_minus_exp(
        -sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf
    )
    if fix_logistic:
        logits = torch.min(logits_1, logits_2)
    else:
        logits = logits_1  # shape before B, C, H, W, S

    return logits


class UViTModel(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        self.S = cfg.data.S
        self.x_min_max = (0, self.S - 1)

        # assert len(cfg.data.shape) == 1

        tmp_net = UViT(
            img_size=cfg.data.image_size,
            num_states=cfg.data.S,
            patch_size=cfg.model.patch_size,
            in_chans=cfg.model.input_channel,
            embed_dim=cfg.model.hidden_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            mlp_time_embed=True,
            num_classes=-1,
            use_checkpoint=False,
            conv=True,
            skip=True,
        ).to(device)
        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.data_shape = cfg.data.shape

    def forward(
        self,
        x: TensorType["B", "D"],
        times: TensorType["B"],
        label: TensorType["B"] = None,
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space
        """
        B, D = x.shape
        x = center_data(x, self.x_min_max)

        logits = self.net(x, times)
        logits = logits.view(B, D, self.S)  # (B, D, S)

        return logits


class DiTModel(nn.Module):
    def __init__(self, cfg, device, rank=None):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.fix_logistic = cfg.model.fix_logistic
        self.S = cfg.data.S
        self.data_shape = cfg.data.shape
        self.model_output = cfg.model.model_output
        self.x_min_max = (0, self.S - 1)

        net = DiT(
            input_size=cfg.data.image_size,  # 28
            num_states=cfg.data.S,
            patch_size=cfg.model.patch_size,  # 2
            in_channels=cfg.model.input_channel,
            hidden_size=cfg.model.hidden_dim,  # 1152
            depth=cfg.model.depth,  # 28
            num_heads=cfg.model.num_heads,  # 16
            mlp_ratio=cfg.model.mlp_ratio,  # 4.0,
            class_dropout_prob=cfg.model.dropout,  # 0.1
            num_classes=self.S,
            model_output=self.model_output,
        )  # logistic_pars output)

        if cfg.distributed:
            self.net = DDP(net, device_ids=[rank])
        else:
            self.net = net

    def forward(
        self, x: TensorType["B", "D"], times: TensorType["B"], y: TensorType["B"] = None
    ) -> TensorType["B", "D", "S"]:
        """
        Returns logits over state space for each pixel
        """
        if len(x.shape) == 2:
            B, D = x.shape
            C, H, W = self.data_shape
            x = x.view(B, C, H, W)
        else:
            B, C, H, W = x.shape

        x = center_data(x, self.x_min_max)
        net_out = self.net(x, times, y)  # (B, 2*C, H, W)

        if self.model_output == "logits":
            out = torch.reshape(net_out, (B, C, self.S, H, W))  # B, C, S, H, W]
            logits = out.permute(0, 1, 3, 4, 2).contiguous()

        elif self.model_output == "logistic_pars":
            loc, log_scale = torch.chunk(net_out, 2, dim=1)
            out = torch.tanh(loc + x), log_scale
            logits = sample_logistic(
                out, B, C, D, self.S, self.fix_logistic, self.device
            )
        else:
            raise ValueError(f"No model output called {self.model_output} registered")
        logits = logits.view(B, D, self.S)
        return logits

# Based on https://github.com/yang-song/score_sde_pytorch/blob/ef5cb679a4897a40d20e94d8d0e2124c3a48fb8c/models/ema.py
class EMA:
    def __init__(self, cfg):
        self.decay = cfg.model.ema_decay
        self.device = cfg.device
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.shadow_params = []
        self.collected_params = []
        self.num_updates = 0

    def init_ema(self):
        self.shadow_params = [
            p.clone().detach() for p in self.parameters() if p.requires_grad
        ]

    def update_ema(self):
        if len(self.shadow_params) == 0:
            raise ValueError("Shadow params not initialized before first ema update!")

        decay = self.decay
        self.num_updates += 1
        decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in self.parameters() if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                param = param.to(self.device)
                s_param = s_param.to(self.device)
                s_param.sub_(one_minus_decay * (s_param - param))

    def state_dict(self):
        sd = nn.Module.state_dict(self)
        sd["ema_decay"] = self.decay
        sd["ema_num_updates"] = self.num_updates
        sd["ema_shadow_params"] = self.shadow_params

        return sd

    def move_shadow_params_to_model_params(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def move_model_params_to_collected_params(self):
        self.collected_params = [param.clone() for param in self.parameters()]

    def move_collected_params_to_model_params(self):
        for c_param, param in zip(self.collected_params, self.parameters()):
            param.data.copy_(c_param.data)

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = nn.Module.load_state_dict(
            self, state_dict, strict=False
        )

        # print("state dict keys")
        # for key in state_dict.keys():
        #     print(key)
        print("ema state dict function")
        if len(missing_keys) > 0:
            print("Missing keys: ", missing_keys)
            raise ValueError
        if not (
            len(unexpected_keys) == 3
            and "ema_decay" in unexpected_keys
            and "ema_num_updates" in unexpected_keys
            and "ema_shadow_params" in unexpected_keys
        ):
            print("Unexpected keys: ", unexpected_keys)
            raise ValueError

        self.decay = state_dict["ema_decay"]
        self.num_updates = state_dict["ema_num_updates"]
        self.shadow_params = state_dict["ema_shadow_params"]

    def train(self, mode=True):
        if self.training == mode:
            print(
                "Dont call model.train() with the same mode twice! Otherwise EMA parameters may overwrite original parameters"
            )
            print("Current model training mode: ", self.training)
            print("Requested training mode: ", mode)
            raise ValueError

        nn.Module.train(self, mode)
        if mode:
            if len(self.collected_params) > 0:
                self.move_collected_params_to_model_params()
            else:
                print("model.train(True) called but no ema collected parameters!")
        else:
            self.move_model_params_to_collected_params()
            self.move_shadow_params_to_model_params()


##############################################################################################################################################################

# make sure EMA inherited first so it can override the state dict functions
# for CIFAR10


@model_utils.register_model
class GaussianUViTEMA(EMA, UViTModel, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        UViTModel.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()


@model_utils.register_model
class GaussianDiTEMA(EMA, DiTModel, GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        DiTModel.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self, cfg, device)

        self.init_ema()
