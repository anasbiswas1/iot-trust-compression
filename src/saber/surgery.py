
"""Physical channel enumeration, masking, FLOP accounting, and surgery for CNN1D.

The implementation targets the simple edge-scale Conv1d -> BatchNorm1d stacks
used by the existing repository. It fails loudly when dependencies cannot be
resolved rather than silently returning a malformed model.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping
import operator

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class PrunableGroup:
    group_id: str
    module_path: str
    layer_index: int
    channel_index: int
    out_channels: int
    in_channels: int
    kernel_size: int
    parameter_cost: int
    flops_cost: float
    batchnorm_path: str | None
    downstream_path: str | None
    downstream_kind: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def get_module(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    if path == "":
        return module
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent_path = ".".join(parts[:-1])
    parent = get_module(root, parent_path) if parent_path else root
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _conv_output_lengths(
    model: nn.Module, example_input: torch.Tensor
) -> dict[str, int]:
    lengths: dict[str, int] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(
                module.register_forward_hook(
                    lambda _m, _i, out, name=name: lengths.__setitem__(
                        name, int(out.shape[-1])
                    )
                )
            )
    training = model.training
    model.eval()
    with torch.no_grad():
        model(example_input)
    model.train(training)
    for hook in hooks:
        hook.remove()
    return lengths


def _module_order(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if name]


def _find_dependencies(
    model: nn.Module, conv_name: str, conv: nn.Conv1d
) -> tuple[str | None, str | None, str | None]:
    ordered = _module_order(model)
    positions = {name: i for i, (name, _) in enumerate(ordered)}
    start = positions[conv_name]
    bn_path: str | None = None
    downstream_path: str | None = None
    downstream_kind: str | None = None

    # Pair the first compatible BN after this convolution before another Conv1d.
    for name, module in ordered[start + 1 :]:
        if isinstance(module, nn.Conv1d):
            if module.in_channels == conv.out_channels:
                downstream_path = name
                downstream_kind = "conv_in"
            break
        if (
            bn_path is None
            and isinstance(module, nn.BatchNorm1d)
            and module.num_features == conv.out_channels
        ):
            bn_path = name

    if downstream_path is None:
        # The final Conv1d usually feeds an AdaptiveAvgPool1d and a Linear head.
        for name, module in ordered[start + 1 :]:
            if isinstance(module, nn.Linear):
                if module.in_features == conv.out_channels:
                    downstream_path = name
                    downstream_kind = "linear_in"
                    break
                if module.in_features % conv.out_channels == 0:
                    downstream_path = name
                    downstream_kind = "linear_block"
                    break

    return bn_path, downstream_path, downstream_kind


def enumerate_cnn1d_channel_groups(
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    minimum_remaining_per_layer: int = 4,
) -> pd.DataFrame:
    """Enumerate removable output channels and their approximate direct cost."""
    lengths = _conv_output_lengths(model, example_input)
    rows: list[dict[str, object]] = []
    convs = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv1d)
    ]
    if not convs:
        raise ValueError("No Conv1d layers were found.")

    for layer_idx, (name, conv) in enumerate(convs):
        if conv.out_channels <= minimum_remaining_per_layer:
            continue
        bn_path, downstream_path, downstream_kind = _find_dependencies(
            model, name, conv
        )
        length = lengths.get(name, 1)
        kernel = int(conv.kernel_size[0])
        # Multiply-adds are counted as two scalar operations.
        conv_flops = 2.0 * conv.in_channels * kernel * length
        # Removing an output also removes one input from the next Conv/Linear.
        downstream_flops = 0.0
        if downstream_path:
            downstream = get_module(model, downstream_path)
            if downstream_kind == "conv_in" and isinstance(downstream, nn.Conv1d):
                # Output length is retrieved when possible.
                dlen = lengths.get(downstream_path, 1)
                downstream_flops = (
                    2.0
                    * downstream.out_channels
                    * int(downstream.kernel_size[0])
                    * dlen
                )
            elif downstream_kind == "linear_in" and isinstance(downstream, nn.Linear):
                downstream_flops = 2.0 * downstream.out_features
            elif downstream_kind == "linear_block" and isinstance(downstream, nn.Linear):
                block = downstream.in_features // conv.out_channels
                downstream_flops = 2.0 * block * downstream.out_features

        param_cost = conv.in_channels * kernel + (1 if conv.bias is not None else 0)
        if bn_path:
            bn = get_module(model, bn_path)
            if isinstance(bn, nn.BatchNorm1d):
                param_cost += 2  # affine gamma and beta
        if downstream_path:
            downstream = get_module(model, downstream_path)
            if downstream_kind == "conv_in" and isinstance(downstream, nn.Conv1d):
                param_cost += downstream.out_channels * int(downstream.kernel_size[0])
            elif downstream_kind == "linear_in" and isinstance(downstream, nn.Linear):
                param_cost += downstream.out_features
            elif downstream_kind == "linear_block" and isinstance(downstream, nn.Linear):
                block = downstream.in_features // conv.out_channels
                param_cost += block * downstream.out_features

        for channel in range(conv.out_channels):
            rows.append(
                PrunableGroup(
                    group_id=f"{name}:out:{channel}",
                    module_path=name,
                    layer_index=layer_idx,
                    channel_index=channel,
                    out_channels=int(conv.out_channels),
                    in_channels=int(conv.in_channels),
                    kernel_size=kernel,
                    parameter_cost=int(param_cost),
                    flops_cost=float(conv_flops + downstream_flops),
                    batchnorm_path=bn_path,
                    downstream_path=downstream_path,
                    downstream_kind=downstream_kind,
                ).as_dict()
            )
    return pd.DataFrame(rows)


def _new_conv_like(
    old: nn.Conv1d, in_indices: np.ndarray, out_indices: np.ndarray
) -> nn.Conv1d:
    new = nn.Conv1d(
        in_channels=len(in_indices),
        out_channels=len(out_indices),
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    if old.groups != 1:
        raise NotImplementedError(
            "Grouped/depthwise Conv1d surgery is not implemented in the starter pack."
        )
    with torch.no_grad():
        new.weight.copy_(old.weight[out_indices][:, in_indices])
        if old.bias is not None:
            new.bias.copy_(old.bias[out_indices])
    return new


def _new_bn_like(old: nn.BatchNorm1d, keep: np.ndarray) -> nn.BatchNorm1d:
    new = nn.BatchNorm1d(
        num_features=len(keep),
        eps=old.eps,
        momentum=old.momentum,
        affine=old.affine,
        track_running_stats=old.track_running_stats,
        device=old.weight.device if old.affine else old.running_mean.device,
        dtype=old.weight.dtype if old.affine else old.running_mean.dtype,
    )
    with torch.no_grad():
        if old.affine:
            new.weight.copy_(old.weight[keep])
            new.bias.copy_(old.bias[keep])
        if old.track_running_stats:
            new.running_mean.copy_(old.running_mean[keep])
            new.running_var.copy_(old.running_var[keep])
            new.num_batches_tracked.copy_(old.num_batches_tracked)
    return new


def _new_linear_like(old: nn.Linear, in_indices: np.ndarray) -> nn.Linear:
    new = nn.Linear(
        in_features=len(in_indices),
        out_features=old.out_features,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight[:, in_indices])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def prune_cnn1d_channels(
    model: nn.Module,
    prune_map: Mapping[str, Iterable[int]],
    example_input: torch.Tensor,
    *,
    minimum_remaining_per_layer: int = 4,
) -> tuple[nn.Module, pd.DataFrame]:
    """Return a physically smaller copy of a sequential CNN1D model."""
    pruned = deepcopy(model)
    convs = [
        (name, module)
        for name, module in pruned.named_modules()
        if isinstance(module, nn.Conv1d)
    ]
    if not convs:
        raise ValueError("No Conv1d layers found.")

    previous_keep: np.ndarray | None = None
    audit: list[dict[str, object]] = []
    last_conv_name: str | None = None
    last_keep: np.ndarray | None = None

    for name, old_conv in convs:
        remove = sorted({int(i) for i in prune_map.get(name, [])})
        bad = [i for i in remove if i < 0 or i >= old_conv.out_channels]
        if bad:
            raise ValueError(f"Invalid channels for {name}: {bad}")
        keep = np.asarray(
            [i for i in range(old_conv.out_channels) if i not in remove],
            dtype=np.int64,
        )
        if len(keep) < minimum_remaining_per_layer:
            raise ValueError(
                f"{name} would retain {len(keep)} channels; minimum is "
                f"{minimum_remaining_per_layer}."
            )
        in_keep = (
            np.arange(old_conv.in_channels, dtype=np.int64)
            if previous_keep is None
            else previous_keep
        )
        if len(in_keep) != old_conv.in_channels and previous_keep is not None:
            # After replacing the previous layer, the old object in the initially
            # captured list still has its original in_channels. The indices refer
            # to the original tensor, which is correct.
            pass
        bn_path, downstream_path, downstream_kind = _find_dependencies(
            pruned, name, old_conv
        )
        new_conv = _new_conv_like(old_conv, in_keep, keep)
        set_module(pruned, name, new_conv)
        if bn_path:
            old_bn = get_module(pruned, bn_path)
            if isinstance(old_bn, nn.BatchNorm1d):
                set_module(pruned, bn_path, _new_bn_like(old_bn, keep))

        audit.append(
            {
                "module_path": name,
                "old_in_channels": int(old_conv.in_channels),
                "new_in_channels": int(len(in_keep)),
                "old_out_channels": int(old_conv.out_channels),
                "new_out_channels": int(len(keep)),
                "removed": len(remove),
                "removed_indices": ",".join(map(str, remove)),
                "batchnorm_path": bn_path,
                "downstream_path": downstream_path,
                "downstream_kind": downstream_kind,
            }
        )
        previous_keep = keep
        last_conv_name = name
        last_keep = keep

    # Update the first compatible Linear layer after the final Conv1d.
    if last_conv_name is not None and last_keep is not None:
        old_last_conv = dict(convs)[last_conv_name]
        _, downstream_path, downstream_kind = _find_dependencies(
            pruned, last_conv_name, old_last_conv
        )
        if downstream_path:
            linear = get_module(pruned, downstream_path)
            if isinstance(linear, nn.Linear):
                if downstream_kind == "linear_in":
                    linear_keep = last_keep
                elif downstream_kind == "linear_block":
                    block = linear.in_features // old_last_conv.out_channels
                    linear_keep = np.concatenate(
                        [
                            np.arange(i * block, (i + 1) * block, dtype=np.int64)
                            for i in last_keep
                        ]
                    )
                else:
                    linear_keep = None
                if linear_keep is not None:
                    set_module(pruned, downstream_path, _new_linear_like(linear, linear_keep))

    # Validate the physically modified graph.
    training = pruned.training
    pruned.eval()
    with torch.no_grad():
        output = pruned(example_input)
    pruned.train(training)
    if output.ndim != 2:
        raise RuntimeError(f"Pruned model returned unexpected shape {tuple(output.shape)}")

    return pruned, pd.DataFrame(audit)


@contextmanager
def temporarily_zero_group(model: nn.Module, group: Mapping[str, object]):
    """Functionally ablate one structured group and restore it afterwards."""
    path = str(group["module_path"])
    channel = int(group["channel_index"])
    conv = get_module(model, path)
    if not isinstance(conv, nn.Conv1d):
        raise TypeError(f"{path} is not Conv1d")

    tensors: list[tuple[torch.Tensor, torch.Tensor]] = []

    def save_and_zero(tensor: torch.Tensor, index) -> None:
        original = tensor[index].detach().clone()
        tensors.append((tensor[index], original))
        tensor[index].zero_()

    with torch.no_grad():
        save_and_zero(conv.weight, channel)
        if conv.bias is not None:
            save_and_zero(conv.bias, channel)

        bn_path = group.get("batchnorm_path")
        if bn_path:
            bn = get_module(model, str(bn_path))
            if isinstance(bn, nn.BatchNorm1d):
                if bn.affine:
                    save_and_zero(bn.weight, channel)
                    save_and_zero(bn.bias, channel)
                if bn.track_running_stats:
                    # Keep a zero-output channel stable.
                    original_mean = bn.running_mean[channel].detach().clone()
                    original_var = bn.running_var[channel].detach().clone()
                    tensors.append((bn.running_mean[channel], original_mean))
                    tensors.append((bn.running_var[channel], original_var))
                    bn.running_mean[channel].zero_()
                    bn.running_var[channel].fill_(1.0)

        downstream_path = group.get("downstream_path")
        kind = group.get("downstream_kind")
        if downstream_path:
            downstream = get_module(model, str(downstream_path))
            if kind == "conv_in" and isinstance(downstream, nn.Conv1d):
                save_and_zero(downstream.weight, (slice(None), channel))
            elif kind == "linear_in" and isinstance(downstream, nn.Linear):
                save_and_zero(downstream.weight, (slice(None), channel))
            elif kind == "linear_block" and isinstance(downstream, nn.Linear):
                block = downstream.in_features // conv.out_channels
                cols = slice(channel * block, (channel + 1) * block)
                save_and_zero(downstream.weight, (slice(None), cols))

    try:
        yield model
    finally:
        with torch.no_grad():
            for view, original in reversed(tensors):
                view.copy_(original)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_nonzero_parameters(model: nn.Module) -> int:
    return int(sum(torch.count_nonzero(p).item() for p in model.parameters()))


def profile_forward_flops(
    model: nn.Module,
    example_input: torch.Tensor,
) -> dict[str, float]:
    """Profile dense Conv1d/Linear multiply-add FLOPs for one forward pass.

    The result is architecture-level dense compute, not measured latency and not
    a claim about sparse kernels.
    """
    rows: list[dict[str, float | str]] = []
    hooks = []

    def conv_hook(name, module, inputs, output):
        batch = int(output.shape[0])
        out_length = int(output.shape[-1])
        kernel = int(module.kernel_size[0])
        per_item = (
            2.0
            * module.out_channels
            * out_length
            * (module.in_channels / module.groups)
            * kernel
        )
        rows.append(
            {
                "module_path": name,
                "kind": "Conv1d",
                "flops_per_item": float(per_item),
                "batch": batch,
            }
        )

    def linear_hook(name, module, inputs, output):
        batch = int(output.shape[0])
        per_item = 2.0 * module.in_features * module.out_features
        rows.append(
            {
                "module_path": name,
                "kind": "Linear",
                "flops_per_item": float(per_item),
                "batch": batch,
            }
        )

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(
                module.register_forward_hook(
                    lambda m, i, o, name=name: conv_hook(name, m, i, o)
                )
            )
        elif isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    lambda m, i, o, name=name: linear_hook(name, m, i, o)
                )
            )

    training = model.training
    model.eval()
    with torch.no_grad():
        model(example_input)
    model.train(training)
    for hook in hooks:
        hook.remove()

    table = pd.DataFrame(rows)
    return {
        "flops_per_item": float(table["flops_per_item"].sum()) if len(table) else 0.0,
        "n_profiled_modules": int(len(table)),
        "details": table,
    }
