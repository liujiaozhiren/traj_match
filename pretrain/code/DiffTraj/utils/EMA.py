import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. -
                    self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict




class SafeEMA:
    def __init__(self, mu=0.9999, keep_fp32_shadow=True):
        self.mu = float(mu)
        self.keep_fp32_shadow = bool(keep_fp32_shadow)
        self.shadow = {}  # key -> CPU tensor (fp32)

    @staticmethod
    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    @torch.no_grad()
    def register(self, module: nn.Module):
        m = self._unwrap(module)
        self.shadow.clear()
        for k, v in m.state_dict().items():
            # 只对可微/浮点权重与缓冲做 EMA（非浮点/量化/稀疏/元设备跳过）
            if not isinstance(v, torch.Tensor):
                continue
            if not v.dtype.is_floating_point:
                continue
            if getattr(v, "is_meta", False):
                # 未物化权重，跳过（或在外部先 materialize）
                continue
            if v.is_sparse:
                continue
            t = v.detach().cpu()
            if self.keep_fp32_shadow:
                t = t.float()
            self.shadow[k] = t.clone()

    @torch.no_grad()
    def update(self, module: nn.Module):
        m = self._unwrap(module)
        sd = m.state_dict()
        for k, v in sd.items():
            if k not in self.shadow:
                # 新出现的浮点权重，尝试注册；meta/非浮点/稀疏仍跳过
                if (isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                        and not getattr(v, "is_meta", False) and not v.is_sparse):
                    base = v.detach().cpu()
                    if self.keep_fp32_shadow:
                        base = base.float()
                    self.shadow[k] = base.clone()
                else:
                    continue
            # 将 shadow 搬到 v 的设备和 dtype 做更新
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point \
               and not getattr(v, "is_meta", False) and not v.is_sparse:
                sh = self.shadow[k].to(device=v.device, dtype=v.dtype)
                sh = self.mu * sh + (1.0 - self.mu) * v.detach()
                # 放回 CPU/FP32 存
                if self.keep_fp32_shadow:
                    sh = sh.float()
                self.shadow[k] = sh.cpu()

    @torch.no_grad()
    def apply_to(self, module: nn.Module):
        # 用 shadow 覆盖模型（做评测/导出）
        m = self._unwrap(module)
        sd = m.state_dict()
        for k, v in sd.items():
            if k in self.shadow and isinstance(v, torch.Tensor) and v.dtype.is_floating_point \
               and not getattr(v, "is_meta", False) and not v.is_sparse:
                sd[k].copy_(self.shadow[k].to(device=v.device, dtype=v.dtype))
        m.load_state_dict(sd)

    def state_dict(self):
        return {"mu": self.mu, "shadow": self.shadow, "keep_fp32_shadow": self.keep_fp32_shadow}

    def load_state_dict(self, state):
        self.mu = float(state.get("mu", self.mu))
        self.keep_fp32_shadow = bool(state.get("keep_fp32_shadow", self.keep_fp32_shadow))
        self.shadow = state["shadow"]