import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')
from jazz_rnn.utilspy.meters import accuracy
from jazz_rnn.utils.music.vectorXmlConverter import input_2_groups
from jazz_rnn.A_data_prep.gather_data_from_xml import EOS_SYMBOL

OFFSET_TO_5OCT = 72


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb
        base_freq = 10000
        inv_freq = 1 / (base_freq ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq.to(dtype=pos_seq.dtype))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super(ResidualBlock, self).__init__()

        self.d_model = d_model
        self.dropout = dropout

        self.CoreNet = []
        for _ in range(3):
            self.CoreNet += [
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=2),
                nn.Dropout(dropout),
            ]
        self.CoreNet = nn.Sequential(*self.CoreNet)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp):
        #  position-wise feed-forward
        core_out = self.CoreNet(inp)

        #  residual connection + layer normalization
        output = self.layer_norm(inp + core_out)

        return output


class ChordBias(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(ChordBias, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_layer = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(1, 1, 3, padding_mode='circular', padding=2)
        self.weighted_avg = nn.Linear(4, 1, bias=False)
        self.weighted_avg.weight = nn.Parameter(0.25 * torch.ones(1, 4))

        self.CoreNet = nn.Sequential(
            self.conv1, nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            self.weighted_avg,
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, chord):
        qlen, bsz, d_emb = chord.shape
        chord_reshaped = chord.view(qlen * bsz, 4, -1).permute(0, 2, 1).contiguous().view(-1, 1, 4)
        chord_bias = self.CoreNet(chord_reshaped)
        chord_bias = chord_bias.view(qlen, bsz, -1)

        out = inp
        out[:, :, :(d_emb // 4)] += chord_bias
        out = self.layer_norm(out)

        return out


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_head: int, dropout: float, dropatt: float = 0,
                 tgt_len: int = None, ext_len: int = None, mem_len: int = None, pre_lnorm: bool = False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, bsz = w.size(0), w.size(1)
        rlen = r.size(0)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', rw_head_q.to(dtype=w_head_k.dtype), w_head_k)  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        rr_head_q = rr_head_q.to(r_head_k.dtype)
        BD = torch.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 4:
                attn_score = attn_score.float().masked_fill(
                    attn_mask, -float('inf')).type_as(attn_score)
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob.to(dtype=w_head_k.dtype), w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = [w + attn_out]
        else:
            ##### residual connection + layer normalization
            output = [self.layer_norm(w + attn_out)]

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, chord_bias,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.chord_bias = chord_bias

        if chord_bias:
            self.chord_bias = ChordBias(d_model, d_inner, dropout)

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, chords=None):

        if self.chord_bias:
            dec_inp = self.chord_bias(dec_inp, chords)

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = output[0]
        output = [self.pos_ff(output)]

        return output


class MemTransformerLM(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 clamp_len=-1, pitch_sizes=None, duration_sizes=None, offset_sizes=None,
                 converter=None, chord_bias=False):
        super(MemTransformerLM, self).__init__()
        self.converter = converter
        self.chord_bias = chord_bias

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.encode_pitch = nn.Embedding(pitch_sizes[0], pitch_sizes[1], scale_grad_by_freq=True)
        self.encode_duration = nn.Embedding(duration_sizes[0], duration_sizes[1], scale_grad_by_freq=True)
        self.offset = offset_sizes[1] > 0
        if self.offset:
            self.encode_offset = nn.Embedding(offset_sizes[0], offset_sizes[1])

        d_emb_size = pitch_sizes[1]
        d_emb_size += duration_sizes[1]
        if self.offset:
            d_emb_size += offset_sizes[1]

        if not chord_bias:
            d_emb_size += 4 * pitch_sizes[1]

        self.proj_input_needed = d_emb_size != d_model
        if self.proj_input_needed:
            self.input_proj = nn.Linear(d_emb_size, d_model)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, chord_bias=self.chord_bias, pre_lnorm=pre_lnorm)
            )
        self.linear_pitch = nn.Linear(d_model, pitch_sizes[1])
        self.linear_duration = nn.Linear(d_model, duration_sizes[1])

        self.decode_pitch = nn.Linear(pitch_sizes[1], pitch_sizes[0])
        self.decode_duration = nn.Linear(duration_sizes[1], duration_sizes[0])

        if tie_weight:
            self.decode_pitch.weight = self.encode_pitch.weight
            self.decode_duration.weight = self.encode_duration.weight

        reduction = 'mean'
        self.crit = nn.CrossEntropyLoss(reduction=reduction)

        self.clamp_len = clamp_len

        self._create_params()

        self.chord_type_2_hist = nn.Embedding(8, 13)

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, ios_mask, mems=None):
        core_out, hids, mlen, qlen, pos_emb, pos_seq = self.core_forward(dec_inp, ios_mask, mems)

        output_pitch = self.drop(torch.sigmoid(self.linear_pitch(core_out)))
        output_duration = self.drop(torch.sigmoid(self.linear_duration(core_out)))
        decoded_pitch = self.decode_pitch(output_pitch)
        decoded_duration = self.decode_duration(output_duration)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return decoded_pitch, decoded_duration, new_mems, None

    def core_forward(self, dec_inp, ios_mask, mems):
        qlen, bsz, _ = dec_inp.size()
        pitch, duration, offset, root, scale_pitches, chord_pitches, chord_idx, rank, octave = \
            input_2_groups(dec_inp, qlen, bsz)
        pitch_emb = self.encode_pitch(pitch)
        duration_emb = self.encode_duration(duration)
        word_emb = torch.cat((pitch_emb, duration_emb), 2)
        if self.offset:
            offset_emb = self.encode_offset(offset)
        else:
            word_emb = torch.cat((pitch_emb, duration_emb), 2)
        if self.offset:
            word_emb = torch.cat((word_emb, offset_emb), 2)

        eos_mask = pitch == EOS_SYMBOL
        chord_emb = self.get_chord_pitch_emb(bsz, qlen, chord_pitches, eos_mask)
        if not self.chord_bias:
            word_emb = torch.cat((word_emb, chord_emb), 2)
        if self.proj_input_needed:
            word_emb = self.input_proj(word_emb)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        dec_attn_mask = torch.triu(
            word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]
        dec_attn_mask = dec_attn_mask.unsqueeze(-1).expand(qlen, klen, bsz, 1)
        for b, seq_idx in ios_mask:
            dec_attn_mask[seq_idx:, :seq_idx, b, :] = 1

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids = []
        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            layer_out = layer(core_out, pos_emb, self.r_w_bias,
                              self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i,
                              chords=chord_emb)
            core_out = layer_out[0]
            hids.append(core_out)
        core_out = self.drop(core_out)

        return core_out, hids, mlen, qlen, pos_emb, pos_seq

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)
        eos_mask = (target[:, :, 0] == 129).t().nonzero()
        decoded_pitch, decoded_duration, new_mems, _ = self._forward(data, eos_mask, mems=mems)

        pitch_logits = decoded_pitch[-tgt_len:]
        duration_logits = decoded_duration[-tgt_len:]
        pitch_logits = pitch_logits.view(-1, pitch_logits.shape[-1])
        duration_logits = duration_logits.view(-1, duration_logits.shape[-1])
        pitch_loss = self.crit(pitch_logits, target[:, :, 0].view(-1))
        duration_loss = self.crit(duration_logits, target[:, :, 1].view(-1))

        qlen, bsz, _ = data.size()
        pitch_probs = F.softmax(pitch_logits, -1)

        nll_loss = pitch_loss + duration_loss

        loss = nll_loss

        pitch_entropy = torch.distributions.categorical.Categorical(logits=pitch_probs).entropy().mean()
        duration_entropy = torch.distributions.categorical.Categorical(logits=duration_logits).entropy().mean()
        total_entropy = pitch_entropy + duration_entropy
        pitch_top1, pitch_top3, pitch_top5 = accuracy(pitch_logits, target.view(-1, 31)[:, 0].contiguous(),
                                                      topk=(1, 3, 5))
        duration_top1, duration_top3 = accuracy(duration_logits, target.view(-1, 31)[:, 1].contiguous(), topk=(1, 3))

        loss_dict = {'loss': loss, 'nll': nll_loss,
                     'p_nll': pitch_loss, 'd_nll': duration_loss,
                     'p_entropy': pitch_entropy, 'd_entropy': duration_entropy,
                     't_entropy': total_entropy,
                     'p_top1': pitch_top1, 'p_top3': pitch_top3, 'p_top5': pitch_top5,
                     'd_top1': duration_top1, 'd_top3': duration_top3}

        loss_list = [loss] if new_mems is None else [loss] + new_mems

        return (pitch_logits, duration_logits), loss_list, loss_dict, None

    def get_chord_pitch_emb(self, batch_size, bptt, chord_pitches, eos_mask):
        shape = chord_pitches.shape

        # eos has no chord, so we need to mask it out
        eos_mask = eos_mask.view(-1)
        eos_mask = 1 - eos_mask
        chord_pitches = chord_pitches.view(-1, 13)
        chord_pitch_idxs_no_eos = chord_pitches.nonzero()[:, 1] + OFFSET_TO_5OCT
        chord_pitch_idxs = torch.zeros(shape[0] * shape[1], 4, device=chord_pitches.device, dtype=chord_pitches.dtype)
        try:
            chord_pitch_idxs[eos_mask, :] = chord_pitch_idxs_no_eos.view(-1, 4)
        except RuntimeError:
            pass
        chord_pitches_emb = self.encode_pitch(chord_pitch_idxs).view(bptt, batch_size, -1).contiguous()
        return chord_pitches_emb
