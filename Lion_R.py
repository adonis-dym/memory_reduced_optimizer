# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch implementation of the Lion optimizer."""
import torch
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt,
                                   _stack_if_compiling, _capturable_doc, _differentiable_doc, _foreach_doc,
                                   _fused_doc, _maximize_doc, _default_to_fused_or_foreach, ParamsT, _view_as_real)
from typing import List
from torch import Tensor


class Lion_R(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self,
                 params,
                 lr=1e-4,
                 betas=(0.9, 0.99),
                 weight_decay=0.0,
                 foreach: bool = True
                 ):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            foreach=foreach
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Lion_R, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avgs.append(state['exp_avg'])
            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                lr=group['lr'],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group['weight_decay'],
            )
            if group['foreach']:
                self._multi_tensor_lion(**kwargs)
            else:
                self._single_tensor_lion(**kwargs)
        return loss

    def _multi_tensor_lion(self,
                           params: List[Tensor],
                           grads: List[Tensor],
                           exp_avgs: List[Tensor],
                           lr: float,
                           beta1: float,
                           beta2: float,
                           weight_decay: float
                           ):
        # weight decay
        torch._foreach_mul_(params, 1 - lr * weight_decay)

        torch._foreach_lerp_(exp_avgs, grads, 1 - beta2)
        torch._foreach_lerp_(grads, exp_avgs, beta1/beta2)
        torch._foreach_sign_(grads)
        torch._foreach_add_(params, grads, alpha=-lr)
