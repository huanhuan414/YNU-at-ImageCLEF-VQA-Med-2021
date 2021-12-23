import torch
import torch.nn as nn
import torch.nn.functional as F
from fc import MLP

# -------------------------------------------------------------
# ---- Multi-Model Hign-order Bilinear Pooling Co-Attention----
# -------------------------------------------------------------

class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size,is_first):
        super(MFB, self).__init__()
        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, 5 * 1000)
        self.proj_q = nn.Linear(ques_feat_size, 5 * 1000)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AvgPool1d(5, stride=5)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)  # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)  # (N, 1, K*O)

        exp_out = img_feat * ques_feat  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)  # (N, C, K*O)
        z = self.pool(exp_out) * 5  # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))  # (N, C*O)
        z = z.view(batch_size, -1, 1000)  # (N, C, O)
        return z, exp_out

class QAtt(nn.Module):
    def __init__(self):
        super(QAtt, self).__init__()
        self.mlp = MLP(
            in_size=768,
            mid_size=512,
            out_size=2,
            dropout_r=0.3,
            use_relu=True
        )

    def forward(self, ques_feat):
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # (N, T, Q_GLIMPSES)
        qatt_maps = F.softmax(qatt_maps, dim=1)         # (N, T, Q_GLIMPSES)

        qatt_feat_list = []
        for i in range(2):
            mask = qatt_maps[:, :, i:i + 1]             # (N, T, 1)
            mask = mask * ques_feat                     # (N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, LSTM_OUT_SIZE)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # (N, LSTM_OUT_SIZE*Q_GLIMPSES)

        return qatt_feat


class IAtt(nn.Module):
    def __init__(self,img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.mfb = MFB(img_feat_size, ques_att_feat_size, True)
        self.mlp = MLP(
            in_size=1000,
            mid_size=512,
            out_size=2,
            dropout_r=0.3,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)      # (N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)
        z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)

        iatt_maps = self.mlp(z)                         # (N, C, I_GLIMPSES)
        iatt_maps = F.softmax(iatt_maps, dim=1)         # (N, C, I_GLIMPSES)

        iatt_feat_list = []
        for i in range(2):
            mask = iatt_maps[:, :, i:i + 1]             # (N, C, 1)
            mask = mask * img_feat                      # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        return iatt_feat

class CoAtt(nn.Module):
    def __init__(self):
        super(CoAtt, self).__init__()
        img_feat_size = 1472
        img_att_feat_size = img_feat_size * 2
        ques_att_feat_size = 768 * 2

        self.q_att = QAtt()
        self.i_att = IAtt(img_feat_size, ques_att_feat_size)
        # MFH
        self.mfh1 = MFB(img_att_feat_size, ques_att_feat_size, True)
        self.mfh2 = MFB(img_att_feat_size, ques_att_feat_size, False)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)
        # MFH
        z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
        z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
        z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)

        return z
