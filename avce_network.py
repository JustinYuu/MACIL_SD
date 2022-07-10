import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from Transformer import *
import copy


class AVCE_Model(nn.Module):
    def __init__(self, args):
        super(AVCE_Model, self).__init__()
        c = copy.deepcopy
        dropout = args.dropout
        nhead = args.nhead
        hid_dim = args.hid_dim
        ffn_dim = args.ffn_dim
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        self.fc_v = nn.Linear(1024, hid_dim)
        self.fc_a = nn.Linear(128, hid_dim)
        self.cma = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        self.att_mmil = Att_MMIL(hid_dim, args.num_classes)

    def forward(self, f_a, f_v, seq_len):
        f_v, f_a = self.fc_v(f_v), self.fc_a(f_a)
        v_out, a_out = self.cma(f_v, f_a)
        mmil_logits, audio_logits, visual_logits, av_logits = self.att_mmil(a_out, v_out, seq_len)
        return mmil_logits, audio_logits, visual_logits, av_logits, v_out, a_out


class Att_MMIL(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Att_MMIL, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def clas(self, logits, seq_len):
        logits = logits.squeeze()
        instance_logits = torch.zeros(0).cuda()  # tensor([])
        for i in range(logits.shape[0]):
            if seq_len is None:
                tmp = torch.mean(logits[i]).view(1)
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, a_out, v_out, seq_len):
        # prediction
        x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2)
        frame_prob = self.fc(x)
        av_logits = frame_prob.sum(dim=2)
        a_logits = torch.sigmoid(frame_prob[:, :, 0, :])
        v_logits = torch.sigmoid(frame_prob[:, :, 1, :])
        mmil_logits = self.clas(av_logits, seq_len)
        return mmil_logits, a_logits, v_logits, av_logits


class Single_Model(nn.Module):
    def __init__(self, args):
        super(Single_Model, self).__init__()
        c = copy.deepcopy
        dropout = args.dropout
        nhead = args.nhead
        hid_dim = args.hid_dim
        ffn_dim = args.ffn_dim
        n_dim = args.v_feature_size
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        self.fc_v = nn.Linear(n_dim, hid_dim)
        self.cma = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        self.fc = nn.Linear(hid_dim, args.num_classes)

    def clas(self, logits, seq_len):
        logits = logits.squeeze()
        instance_logits = torch.zeros(0).cuda()  # tensor([])
        for i in range(logits.shape[0]):
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, f, seq_len):
        f = self.fc_v(f)
        sa = self.cma(f)
        out = self.fc(sa)
        if seq_len is not None:
            out = self.clas(out, seq_len)
        return out
