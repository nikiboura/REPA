"""
Weight selection initialization for SiT models.
Based on: "Initializing Models with Larger Ones" (arxiv 2311.18823)

Transfers weights from a larger pretrained SiT checkpoint to a smaller SiT model
by uniformly selecting a subset of weights across every dimension.

Encoder blocks are mapped only to teacher encoder blocks, and decoder blocks
only to teacher decoder blocks, respecting the functional role of each block.
"""

import torch
from utils import load_legacy_checkpoints


def uniform_element_selection(wt, s_shape):
    """Select a subset of wt to match s_shape by uniform sampling along each dim."""
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        if wt.shape[dim] == s_shape[dim]:
            continue
        assert wt.shape[dim] >= s_shape[dim], \
            f"Teacher dim {wt.shape[dim]} < student dim {s_shape[dim]} on dim {dim}"
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim] - 1, s_shape[dim])).long()
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == torch.Size(s_shape)
    return ws


def _build_block_map(s_enc, s_dec, t_enc, t_dec):
    """
    Build a per-block mapping from student index → teacher index.

    Encoder blocks map within [0, t_enc) and decoder blocks map within
    [t_enc, t_enc+t_dec), keeping the two groups independent.
    """
    enc_map = torch.round(torch.linspace(0, t_enc - 1, s_enc)).long().tolist()
    dec_offsets = torch.round(torch.linspace(0, t_dec - 1, s_dec)).long().tolist()
    dec_map = [t_enc + o for o in dec_offsets]
    return enc_map + dec_map


def weight_selection_init(student_model, teacher_ckpt_path):
    """
    Initialize student_model weights from a larger teacher checkpoint.

    Compatible layers are initialized via weight selection.
    Incompatible layers (patch_embed, pos_embed, projectors) keep their
    default random/sin-cos initialization.

    Args:
        student_model: SiT model instance (already constructed)
        teacher_ckpt_path: path to teacher .pt checkpoint (expects 'ema' key)

    Returns:
        (selected_keys, skipped_keys)
    """
    ckpt = torch.load(teacher_ckpt_path, map_location='cpu', weights_only=False)
    teacher_sd = ckpt['ema'] if 'ema' in ckpt else ckpt

    # Normalize legacy format (decoder_blocks.X → blocks.{enc_depth+X})
    if any(k.startswith('decoder_blocks.') for k in teacher_sd):
        t_enc_depth = max(int(k.split('.')[1]) for k in teacher_sd if k.startswith('blocks.')) + 1
        teacher_sd = load_legacy_checkpoints(teacher_sd, t_enc_depth)
    else:
        # Infer teacher encoder depth from the checkpoint's own encoder_depth key if present,
        # otherwise assume same encoder_depth as student (safe default for same-config teachers)
        t_enc_depth = teacher_sd.get('encoder_depth', student_model.encoder_depth)
        if isinstance(t_enc_depth, torch.Tensor):
            t_enc_depth = int(t_enc_depth)

    student_sd = student_model.state_dict()
    s_enc_depth = student_model.encoder_depth

    t_total = max(int(k.split('.')[1]) for k in teacher_sd if k.startswith('blocks.')) + 1
    s_total = max(int(k.split('.')[1]) for k in student_sd if k.startswith('blocks.')) + 1

    t_dec_depth = t_total - t_enc_depth
    s_dec_depth = s_total - s_enc_depth

    block_map = _build_block_map(s_enc_depth, s_dec_depth, t_enc_depth, t_dec_depth)

    new_sd = {}
    selected, skipped = [], []

    for s_key, s_param in student_sd.items():
        # Remap block index using the role-aware map
        if s_key.startswith('blocks.'):
            parts = s_key.split('.')
            t_block_idx = block_map[int(parts[1])]
            t_key = 'blocks.' + str(t_block_idx) + '.' + '.'.join(parts[2:])
        else:
            t_key = s_key

        if t_key not in teacher_sd:
            skipped.append(s_key)
            continue

        t_param = teacher_sd[t_key]
        s_shape = list(s_param.shape)

        # Skip if teacher is smaller than student in any dim (can't select)
        if any(t_param.shape[d] < s_shape[d] for d in range(t_param.dim())):
            skipped.append(s_key)
            continue

        # Skip pos_embed and x_embedder (sin-cos / incompatible patch size)
        if s_key in ('pos_embed', 'x_embedder.proj.weight', 'x_embedder.proj.bias'):
            skipped.append(s_key)
            continue

        # Skip projectors (z_dim depends on encoder, not teacher)
        if s_key.startswith('projectors'):
            skipped.append(s_key)
            continue

        try:
            new_sd[s_key] = uniform_element_selection(t_param, s_shape)
            selected.append(s_key)
        except Exception as e:
            skipped.append(s_key)

    missing, unexpected = student_model.load_state_dict(new_sd, strict=False)

    enc_map = list(enumerate(block_map[:s_enc_depth]))
    dec_map = list(enumerate(block_map[s_enc_depth:], start=s_enc_depth))
    print(f'Weight selection: {len(selected)} layers initialized from teacher')
    print(f'Skipped (kept default init): {len(skipped)} layers')
    print(f'Encoder block mapping (student→teacher): {enc_map}')
    print(f'Decoder block mapping (student→teacher): {dec_map}')

    return selected, skipped
