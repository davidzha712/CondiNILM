"""Mixture of Experts (MoE) layers for Switch Transformer routing.

Provides a FeedForward expert module and a SwitchFeedForward router that
dispatches tokens to the highest-probability expert. Used by STNILM.
"""
from typing import Any, TypeVar, Iterator, Iterable, Generic

import torch.nn
import torch.nn as nn
import copy


class Module(torch.nn.Module):
    """Base module that redirects __call__ to forward for better type checking."""

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get("__call__", None) is None:
            return

        setattr(cls, "forward", cls.__dict__["__call__"])
        delattr(cls, "__call__")

    @property
    def device(self):
        params = self.parameters()
        try:
            sample_param = next(params)
            return sample_param.device
        except StopIteration:
            raise RuntimeError(
                f"Unable to determine device of {self.__class__.__name__}"
            ) from None


M = TypeVar("M", bound=torch.nn.Module)
T = TypeVar("T")


class TypedModuleList(torch.nn.ModuleList, Generic[M]):
    def __getitem__(self, idx: int) -> M:
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: M) -> None:
        return super().__setitem__(idx, module)

    def __iter__(self) -> Iterator[M]:
        return super().__iter__()

    def __iadd__(self: T, modules: Iterable[M]) -> T:
        return super().__iadd__(modules)

    def insert(self, index: int, module: M) -> None:
        super().insert(index, module)

    def append(self: T, module: M) -> T:
        return super().append(module)

    def extend(self: T, modules: Iterable[M]) -> T:
        return super().extend(modules)

    def forward(self):
        raise NotImplementedError()


def clone_module_list(module: M, n: int) -> TypedModuleList[M]:
    """Create a ModuleList of n independent deep copies of module."""
    return TypedModuleList([copy.deepcopy(module) for _ in range(n)])


class FeedForward(Module):
    """Position-wise feed-forward network with optional gating.

    Computes: Linear -> activation -> (optional gate) -> dropout -> Linear.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)


class SwitchFeedForward(Module):
    """Switch Transformer routing layer: dispatches each token to the top-1 expert."""

    def __init__(
        self,
        d_model,
        expert,
        capacity_factor=1.2,
        drop_tokens=False,
        is_scale_prob=False,
        n_experts=4,
    ):
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        self.experts = clone_module_list(expert, n_experts)
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """Route tokens to experts.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            Tuple of (output, counts, route_prob_sum, n_dropped, route_prob_max)
            for load-balancing loss computation.
        """
        seq_len, batch_size, d_model = x.shape
        x = x.view(-1, d_model)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        indexes_list = [
            torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)
        ]

        final_output = x.new_zeros(x.shape)
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) <= capacity:
                    continue
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                dropped.append(indexes_list[i][capacity:])
                indexes_list[i] = indexes_list[i][:capacity]

        expert_output = [
            self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)
        ]

        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Gradient pass-through: multiply by p/p.detach() = 1 to keep gradients flowing
            final_output = final_output * (
                route_prob_max / route_prob_max.detach()
            ).view(-1, 1)

        final_output = final_output.view(seq_len, batch_size, d_model)

        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max
