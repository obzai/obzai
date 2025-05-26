# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Sequence, Literal, Union, Optional
from captum.attr import Saliency, NoiseTunnel
from abc import ABC, abstractmethod
from torch import nn
import numpy as np
import contextlib
import torch


class XAITool(ABC):
    """
    Base class for each of XAI tool.
    Each XAI method must have .explain() method which hadles batch of input images!
    """
    def __init__(self):
        pass


class GradientTool(XAITool):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def explain(self, batch: torch.Tensor, target_idx: Union[int, Sequence[int]]) -> torch.Tensor:
        pass


class AttentionTool(XAITool):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def explain(self, batch: torch.Tensor):
        pass


# Gradients based tools:

class CDAM(GradientTool):
    def __init__(self, 
                 model: nn.Module, 
                 mode: Literal["vanilla", "smooth", "integrated"] = "vanilla",
                 gradient_type: Literal["from_logits", "from_probabilities"]="from_logits",
                 gradient_reduction: Literal["dot_product", "average", "sum"]="dot_product",
                 activation_type: Literal["sigmoid", "softmax"]="softmax",
                 noise_level: float = 0.05,
                 num_steps: int = 50,
                 ommit_tokens: int = 1
                 ):
        super().__init__()
        self.id = 1
        self.mode = mode
        self.model = model.eval()
        self.gradient_type = gradient_type
        self.gradient_reduction = gradient_reduction
        self.activation_type = activation_type
        self.noise_level = noise_level
        self.num_steps = num_steps
        self.ommit_tokens = ommit_tokens

        self.gradients = {}
        self.activations = {}
        self.created_hooks = False
        self.run_hook = False

    @contextlib.contextmanager
    def hook_manager(self):
        """Context manager to enable and disable hooks automatically."""
        self.run_hook = True
        try:
            yield
        finally:
            self.run_hook = False
    
    def create_hooks(self, layer_name: str):
        """Sets up forward and backward hooks on the specified model layer."""
        if self.created_hooks:
            raise RuntimeError("Hooks already exist. Remove them before creating new ones.")

        self.layer_name = layer_name
        layer = dict(self.model.named_modules())[layer_name]

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if self.run_hook:
                    #print("Backward hook runs on the module: ", module)
                    self.gradients[name] = grad_output[0]
            return hook
        
        def forward_hook(name):
            def hook(module, input, output):
                if self.run_hook:
                    #print("Forward hook runs on the module: ", module)
                    if self.mode == "vanilla":
                        self.activations[name] = output.detach()
                        return output
                    elif self.mode == "smooth":
                        std = self.noise_level * (output.max() - output.min()).item()
                        noise = torch.normal(0.0, std, output.shape, device=output.device)
                        modified_output = output + noise.requires_grad_()
                        self.activations[name] = modified_output.detach()
                        return modified_output
                    elif self.mode == "integrated":
                        baseline = torch.zeros_like(output, device=output.device).requires_grad_()
                        std = self.noise_level * (output.max() - output.min()).item()
                        noise = torch.normal(0.0, std, output.shape, device=output.device)
                        alpha = self.iteration / self.num_steps
                        modified_output = baseline + alpha * (output - baseline) + noise
                        self.activations[name] = modified_output.detach()
                        return modified_output
            return hook

        self.gradient_hook = layer.register_full_backward_hook(backward_hook(layer_name))
        self.activation_hook = layer.register_forward_hook(forward_hook(layer_name))
        self.created_hooks = True
    
    def remove_hooks(self):
        """Removes hooks if they exist."""
        if self.created_hooks:
            self.gradient_hook.remove()
            self.activation_hook.remove()
            self.created_hooks = False
        else:
            raise RuntimeWarning("No hooks to remove.")

    def _compute_cdam(self, batch: torch.Tensor, target_idx: Union[int, Sequence[int]]):
        """
        We can also get inspired with Captum way of gradient computations:
        Look a the function: def compute_gradients(...) at
        https://github.com/pytorch/captum/blob/master/captum/_utils/gradient.py#L103
        I like their way to compute gradients without calling .backward() explicite.
        """
        B, C, H, W = batch.shape
        target_idx = [target_idx] * B if isinstance(target_idx, int) else target_idx
        
        with self.hook_manager():
            outputs = self.model(batch)
            self.model.zero_grad()

            if self.gradient_type == "from_logits":
                outputs[range(B), target_idx].sum().backward()
            elif self.gradient_type == "from_probabilities":
                if self.activation_type == "sigmoid":
                    probs = torch.nn.functional.sigmoid(outputs)
                elif self.activation_type == "softmax":
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                else:
                    raise ValueError(f"Provided {self.activation_type} is not a valid activation method!")
                probs[range(B), target_idx].sum().backward()
            else:
                raise ValueError(f"Provided {self.gradient_type} is not a valid gradient method!")
        
        tokens = self.activations[self.layer_name][:, self.ommit_tokens:, :]
        grads = self.gradients[self.layer_name][:, self.ommit_tokens:, :]

        if self.gradient_reduction == "dot_product":
            cdam_scores = torch.einsum('bij,bij->bi', tokens, grads).reshape(B, int(np.sqrt(grads.size(1))), -1)
        elif self.gradient_reduction == "average":
            cdam_scores = torch.mean(grads, dim=2).reshape(B, int(np.sqrt(grads.size(1))), -1)
        elif self.gradient_reduction == "sum":
            cdam_scores = torch.sum(grads, dim=2).reshape(B, int(np.sqrt(grads.size(1))), -1)
        else:
            raise ValueError(f"Provided {self.gradient_reduction} is not a valid gradient reduction method!")

        cdam_scores = torch.clamp(cdam_scores, min=torch.quantile(cdam_scores, 0.01), max=torch.quantile(cdam_scores, 0.99))
        return torch.nn.functional.interpolate(cdam_scores.unsqueeze(dim=1), scale_factor=H / cdam_scores.size(1), mode="nearest").cpu()

    def explain(self, batch: torch.Tensor, target_idx: Union[int, Sequence[int]]) -> torch.Tensor:
        if not self.created_hooks:
            raise RuntimeError("Hooks must be created before calling .explain()")
        
        def compute_average_maps(num_iterations: int) -> torch.Tensor:
            return torch.mean(torch.stack([self._compute_cdam(batch, target_idx) for self.iteration in range(num_iterations)]), dim=0)
        
        if self.mode == "vanilla":
            return self._compute_cdam(batch, target_idx)
        elif self.mode in {"smooth", "integrated"}:
            return compute_average_maps(self.num_steps + (1 if self.mode == "integrated" else 0))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class SaliencyMap(XAITool):
    def __init__(self, 
                 model: nn.Module,
                 mode: Literal["vanilla", "smooth"] = "vanilla",
                 nt_type: Literal["smoothgrad", "smoothgrad_sq", "vargrad"] = "smoothgrad",
                 nt_samples: int = 10
                 ):
        super().__init__()
        self.id = 2
        self.mode = mode
        self.nt_type = nt_type
        self.nt_samples = nt_samples

        self.model = model.eval()
        self.saliency = Saliency(model)
        if mode == "smooth":
            self.saliency = NoiseTunnel(self.saliency)      
    
    def explain(self, 
                batch: torch.Tensor,
                target_idx: int | Sequence[int]
                ) -> torch.Tensor:
        
        if not batch.requires_grad:
            batch.requires_grad_()
        
        if self.mode == "vanilla":
            smooth_grad = self.saliency.attribute(batch, target=target_idx)
        elif self.mode == "smooth":
            smooth_grad = self.saliency.attribute(batch, nt_type=self.nt_type, nt_samples=self.nt_samples, target=target_idx)
        
        smooth_grad = torch.sum(smooth_grad, dim=1).cpu()
        smooth_grad = torch.clamp(smooth_grad, min=torch.quantile(smooth_grad, 0.01), max=torch.quantile(smooth_grad, 0.99))
        return smooth_grad


# Attention based tools

class AttentionMap(AttentionTool):
    """
    Prepares attention maps.
    Class expects model, which return attention weights, like transformers from huggingface package.
    So, model's forward method must handle keyword argument -> output_attention:bool
    """
    def __init__(self, 
                 model: nn.Module,
                 attention_layer_id:int=-1,
                 head:Optional[int]=None,
                 ommit_tokens:int=1
                 ):
        super().__init__()
        self.id = 3
        self.model = model
        self.attention_layer_id = attention_layer_id
        self.head = head
        self.ommit_tokens = ommit_tokens

    def explain(self, batch: torch.Tensor):
        """
        Provides an attention map for a batch of input images.

        NOTE: target_idx is there only for compatibility with other XAI Tools
        """
        _, _, img_H, img_W = batch.shape
        preds, all_atts = self.model(batch, output_attentions=True)

        if self.attention_layer_id != -1 and self.attention_layer_id not in range(len(all_atts)):
            raise ValueError(f"There are only {len(all_atts)} layers! Please provide layer id: 0 - {len(all_atts)-1}.")

        layer_att = all_atts[self.attention_layer_id].detach()
        B, n_heads, _, _ = layer_att.shape
        att_H = att_W = int(np.sqrt(layer_att.shape[-1] - 1)) # Feature map shape without a CLS token

        # Extracting attentions scores when CLS is used as query
        if self.ommit_tokens > 0:
            layer_att = layer_att[:, :, self.ommit_tokens-1, self.ommit_tokens:].reshape(B, n_heads, -1)
        elif self.ommit_tokens == 0:
            # to handle I-Jepa, which doesn't have CLS token.
            layer_att = layer_att[:, :, 0, :].reshape(B, n_heads, -1)
        else:
            raise ValueError("Ommit tokens argument's value cannot be negative!")
        layer_att = layer_att.reshape(B, n_heads, att_H, att_W)

        layer_att = torch.nn.functional.interpolate(
            layer_att, scale_factor=img_H//att_H, mode="nearest")
        
        if self.head:
            if self.head and self.head not in range(n_heads):
                raise ValueError(f"Provided head argument should be in range: 0-{n_heads-1}.")
            return layer_att[:, self.head, :, :].cpu()
        else:
            averaged_layer_att = torch.mean(layer_att, dim=1).cpu()
            return averaged_layer_att