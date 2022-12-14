from torch import nn
import torch.nn.functional as F
import torch
import itertools
import numpy as np
import os
import json
from tqdm import tqdm
from math import ceil
from transformers import BertConfig,BertModel,BertTokenizer
BERT_PATH = '../chinese_bert/'

torch.autograd.set_detect_anomaly(True)
INF = 1e5
DEFAULT_BATCH_SIZE = 1000
MAX_WINDOW_SIZE = 200

class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        feature = [conv(input_data) for conv in self.convs]
        return feature


class bilstm(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, is_bidirectional, batch_first=True):
        super(bilstm, self).__init__()
        self.batch_first = batch_first
        self.layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=1,
                             bidirectional=is_bidirectional, batch_first=batch_first)

    def forward(self, input, input_length):
        # input = torch.transpose(input, 1, 2)
        output, (h_n, c_n) = self.layer(input)
        return output, (h_n, c_n)

class gate(nn.Module):
    def __init__(self):
        super(gate, self).__init__()
        self.beta = torch.tensor(0.5, requires_grad=False)
    def forward(self, input1, input2):
        output = self.beta * input1 + (1-self.beta) * input2
        return output

class self_attention(nn.Module):
    def __init__(self, seq_length, dropout):
        super(self_attention, self).__init__()
        self.linear = nn.Linear(in_features=seq_length, out_features=1)
        self.dropout = nn.Dropout(p=dropout,inplace=False)

    def forward(self, input):
        x = self.linear(input) #input: [175,100,400] #x: [175, 100, 1]
        point = torch.unsqueeze(torch.sum(x, dim=-1), -1) #point: [175, 100, 1]
        mask = - torch.eq(point, torch.zeros(size=point.shape).cuda()).float() * INF #mask: [175, 100, 1]
        x = x + mask
        p = F.softmax(x, dim=1) #p: [175, 100, 1]
        # [num, max_len, 1]
        c = torch.sum(p * input, dim=1) #c: [175, 400]
        # [num, emb_size]
        c = self.dropout(c) #c: [175, 400]
        return c

class feedforward(nn.Module):
    def __init__(self,
                    input_size,
                    num_layers,
                    num_units,
                    outputs_dim,
                    activation,
                    dropout):
        super(feedforward, self).__init__()
        if activation == 'relu':
            act_type = nn.ReLU(inplace =False)
        else:
            act_type = nn.Tanh(inplace =False)
        layers = []
        if num_layers != 0:
            layers += [nn.Linear(
                in_features=input_size,
                out_features=num_units)
            , act_type]
            layers += [nn.Dropout(p=dropout,inplace =False)]
            for i in range(num_layers - 1):
                layers += [nn.Linear(
                    num_units,
                    num_units
                ), act_type]
                layers += [nn.Dropout(p=dropout,inplace =False)]
            layers += [nn.Linear(num_units, outputs_dim)]
        else:
            layers += [nn.Linear(input_size, outputs_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        output = self.model(input)
        return output

class encoder(nn.Module):
    def __init__(self, is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional, batch_first=True):
        super(encoder, self).__init__()
        self.is_global = is_global
        if is_global:
            self.h_s = bilstm(input_size, hidden_size, dropout, is_bidirectional, batch_first)
            self.h_g = bilstm(input_size, hidden_size, dropout, is_bidirectional, batch_first)
            self.h = gate()
            self.c_s = self_attention(seq_length, dropout)
            self.c_g = self_attention(seq_length, dropout)
            self.c = gate()
        else:
            self.h_g = bilstm(input_size, hidden_size, dropout, is_bidirectional, batch_first)
            self.c = self_attention(hidden_size*2, dropout)

    def forward(self, input, input_length):
        if self.is_global:
            h_s, _ = self.h_s(input, input_length)
            h_g, _ = self.h_g(input, input_length)
            h = self.h(h_s, h_g)
            c_s = self.c_s(h)
            c_g = self.c_g(h)
            c = self.c(c_s, c_g)
        else:
            h, _ = self.h_g(input, input_length) #h: [175,100,400]/[45,35,400]
            c = self.c(h) #c: [175,400]/[45,400]
        return h, c

class U(nn.Module):
    def __init__(self, is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional=True, batch_first=True,
                 window_size=1, max_len=100, num_units=400):
        super(U, self).__init__()
        feature_kernel = {5: 100, 10: 100, 20: 100, 40: 100, 80: 100, 160:100}
        in_size = 100
        self.coder_u = encoder(is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional, batch_first)
        self.coder_c = encoder(is_global, seq_length, input_size, hidden_size, dropout, is_bidirectional, batch_first)
        self.window_size = window_size
        self.max_len = max_len
        self.num_units = num_units

    def forward(self, uinput, uinput_length, cinput, cinput_length):
        utt_h, _ = self.coder_u(uinput, uinput_length) #uinput: [175,100,300] #cinput: [45(ontology key?????????), 35(max length), 300]
        # [batch_size * window_size, max_len, num_units]
        utt_h = torch.reshape(utt_h, [-1, self.window_size, self.max_len, self.num_units])
        # [batch_size, window_size, max_len, num_units]
        _, candidate_c = self.coder_c(cinput, cinput_length)
        # [slot_value_num, num_units]
        return utt_h, candidate_c

class D(nn.Module):
    def __init__(self, num_units, num_layers,
                    outputs_dim,
                    activation,
                    dropout):
        super(D, self).__init__()
        self.num_units = num_units
        self.weight = torch.empty(size=[1, num_units, num_units],
                dtype=torch.float32, requires_grad=True)
        nn.init.xavier_normal_(self.weight)
        self.weight = nn.parameter.Parameter(self.weight)
        self.feed_forward = feedforward(2 * num_units, num_layers,
                    num_units,
                    outputs_dim,
                    activation,
                    dropout)
        self.a = None

    def _attention(self,
                  query,  # [slot_value_num, num_units]
                  keys,  # [batch_size, window_size, max_len, num_units]
                  values  # [batch_size, window_size, max_len, num_units]
                  ):
        self.batch_size, self.window_size = keys.shape[0], keys.shape[1]
                #wtf?
        query = torch.unsqueeze(torch.unsqueeze(query, 0), 0).repeat(self.batch_size, self.window_size, 1, 1)
        # [batch_size, window_size, slot_value_num, num_units]
        p = torch.matmul(
            query,
            torch.transpose(keys, 2, 3)  # [batch_size, window_size, num_units, max_len]
        )  # [batch_size, window_size, slot_value_num, max_len]
        #wtf?
        mask = - torch.mul(torch.eq(p, torch.zeros(p.shape).cuda()).float(), INF)
        p = F.softmax(p + mask, dim=-1)

        outputs = torch.matmul(p, values)
        # [batch_size, window_size, slot_value_num, num_units]
        return outputs

    def forward(self, slot_utt_h, slot_candidate_c, status_candidate_c, position_encoding, q_status, mask):
        q_slot = self._attention(slot_candidate_c, slot_utt_h, slot_utt_h) \
                 + position_encoding  #slot_candicate_c: [slot_value_num, num_units] slot_utt_h: [batch_size, window_size, max_len, num_units]
        slot_value_num = slot_candidate_c.shape[0]
        status_num = status_candidate_c.shape[0]
      # [batch_size, num_units, num_units]
        #?????????eq 6
        co = torch.reshape(
            torch.reshape(
                torch.matmul(
                    torch.matmul(
                        torch.reshape(
                            q_slot,
                            [self.batch_size, self.window_size * slot_value_num, self.num_units]
                        ),
                        self.weight.repeat([self.batch_size, 1, 1]).to('cuda')
                    ),
                    torch.reshape(
                        q_status,
                        [self.batch_size, self.window_size * status_num, self.num_units]
                    ).transpose(1, 2)
                ),  # [batcn_size, window_size * slot_value_num, window_size * status_num]
                [self.batch_size, self.window_size, slot_value_num, self.window_size * status_num]
            ),
            [self.batch_size, self.window_size, slot_value_num, self.window_size, status_num]
        )
        co_mask = - torch.mul(torch.eq(co, torch.zeros(co.shape).cuda()).float(), INF)
        p = co + co_mask

        p = F.softmax(p, 3)
        q_status_slot = torch.unsqueeze(torch.unsqueeze(q_status, -1), -1).repeat(
                [1, 1, 1, 1, self.window_size, slot_value_num])  # [batch_size, window_size, slot_value_num, window_size, status_num, num_units]
        q_status_slot = q_status_slot.transpose(1, 4).transpose(2, 5).transpose(3, 5).transpose(3, 4)
        q_status_slot = torch.sum(torch.mul(
            torch.unsqueeze(p, -1),
            q_status_slot
        ), 3)  # [batch_size, window_size, slot_value_num, status_num, num_units]

        q_slot = torch.unsqueeze(q_slot, 3).repeat([1, 1, 1, status_num, 1])
        features = torch.cat([q_slot, q_status_slot], -1)

        # aggregate
        # [batch_size, window_size, slot_value_num, status_num, 2 * num_units]
        logits = self.feed_forward(features)  # [batch_size, window_size, slot_value_num, status_num, 1]
        logits = torch.reshape(
            logits,
            [-1, self.window_size, slot_value_num * status_num]
        )
        # [batch_size, window_size, slot_value_num * status_num]
        #print(logits.shape)
        slot_pred_logits = torch.max(logits + mask, 1).values
        self.a = slot_pred_logits
        # [batch_size, slot_value_num * status_num]

        #print(slot_pred_logits.shape)
        # ??????slot???????????????
        slot_pred_labels = torch.gt(slot_pred_logits, torch.zeros(slot_pred_logits.shape).cuda()).float()
        # print(slot_pred_labels.grad)
        return slot_pred_labels, slot_value_num * status_num, slot_pred_logits


class ESAL:
    def __init__(self, data, ontology, **kw):
        # ?????????data???ontology
        self.data = data
        self.ontology = ontology
        self.slots = [item[0] for item in self.ontology.ontology_list \
                      if item[0] != self.ontology.mutual_slot]
        # self.weights = [len(values) for _, values in self.ontology.ontology_list]
        self.bert = BertModel.from_pretrained('../pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
        self.max_len = self.data.max_len
        self.params = kw['params']
        self.device = torch.device('cuda:{}'.format(self.params['gpu_ids'])) if self.params['gpu_ids'] else torch.device('cpu')
        self.dropout = 1 - self.params['keep_p']
        self.batch_size = self.params['batch_size']
        self.Umodel_names = [item[0] for item in self.ontology.ontology_list] #['D_A', 'D_B', 'D_C', 'D_D']
        self.Dmodel_names = self.slots
        self.bert.to(self.device)
        self.num_units = self.params['num_units']
        self.init()
        self.Umodels = {}
        self.Dmodels = {}
        self.opt = {}
        for item in self.Umodel_names:
            self.Umodels[item] = U(is_global=self.params['add_global'],input_size=768, seq_length=self.max_len,hidden_size=int(self.num_units/2), dropout=self.dropout)
            self.Umodels[item].to(self.device)
        for item in self.slots:
            self.Dmodels[item] = D(num_units=self.num_units,
                                   num_layers=self.params['num_layers'], outputs_dim=1, activation='relu', dropout=self.dropout)
            self.Dmodels[item].to(self.device)
        for item in self.Dmodel_names:
            self.opt[item] = torch.optim.Adam(itertools.chain(self.Umodels[item].parameters(), self.Dmodels[item].parameters()), lr=self.params['lr'])
        self.mutual_opt = torch.optim.Adam(self.Umodels[self.ontology.mutual_slot].parameters())
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion1 = nn.BCEWithLogitsLoss(reduction='none')
        self.reinit()
        # for item in self.Dmodel_names:
        #     print(self.opt[item].param_groups)

    def set_eval_mode(self):
        for item in self.Dmodel_names:
            self.Dmodels[item].eval()
        for item in self.Umodel_names:
            self.Umodels[item].eval()
    
    def set_train_mode(self):
        for item in self.Dmodel_names:
            self.Dmodels[item].train()
        for item in self.Umodel_names:
            self.Umodels[item].train()

    def set_window_size(self,windows_utts_batch, eval=False, win_size = 5):
        if eval:
            self.window_size = win_size
            for item in self.Umodels:
                self.Umodels[item].window_size = win_size
        else:
            self.window_size = windows_utts_batch.shape[1]
            for item in self.Umodels:
                self.Umodels[item].window_size = self.window_size

    def init(self):
        self.windows_utts = torch.Tensor()
        # [batch_size * window_size]
        self.slots_pred_labels = []

    def set_input(self, windows_utts, windows_utts_lens, labels):
        self.windows_utts = torch.tensor(windows_utts).long().to(self.device)
        self.windows_utts_lens = torch.tensor(windows_utts_lens).to(self.device)
        self.labels = labels
        self.slot_utt_hs_dict = dict()
        self.slot_candidate_cs_dict = dict()
        self.candidate_seqs_dict, self.candidate_seqs_lens_dict = self.ontology.onto2ids()
        # to tensor
        for slot in self.candidate_seqs_dict.keys():
            self.candidate_seqs_dict[slot] = torch.tensor(self.candidate_seqs_dict[slot]).long().to(self.device)
            self.candidate_seqs_lens_dict[slot] = torch.tensor(self.candidate_seqs_lens_dict[slot]).to(self.device)

    def _get_embedding(self):
        for slot in self.candidate_seqs_dict.keys():
            self.candidate_seqs_dict[slot] = self.candidate_seqs_dict[slot]
        # [batch_size, window_size, max_len, emb_size]
        # dim = tf.reduce_prod(tf.shape(windows_utts[:2]))
        self.utts = torch.reshape(self.windows_utts, [-1, self.max_len])
        self.utts_lens = torch.reshape(self.windows_utts_lens, [-1])#[]


    def _attention(self,
                  query,  # [slot_value_num, num_units]
                  keys,  # [batch_size, window_size, max_len, num_units]
                  values  # [batch_size, window_size, max_len, num_units]
                  ):
        batch_size, window_size = keys.shape[0], keys.shape[1]
        query = torch.unsqueeze(torch.unsqueeze(query, 0), 0).repeat(batch_size, window_size, 1, 1) #query: [slot_value_num,num_units] -> [batch_size, window_size, slot_value_num,num_units]

        # [batch_size, window_size, slot_value_num, num_units]
        # print(query.shape, keys.shape)
        p = torch.matmul(
            query,
            keys.transpose(2, 3)  # [batch_size, window_size, max_len, num_units] - > [batch_size, window_size, num_units, max_len]
        )  # [batch_size, window_size, slot_value_num, max_len]

        mask = -torch.eq(p, torch.zeros(p.shape).cuda()).float() * INF
        p = F.softmax(p + mask, dim=-1)

        outputs = torch.matmul(p, values)
        # [batch_size, window_size, slot_value_num, num_units]

        return outputs

    def _position_encoding(self):
        num_units = self.params['num_units']
        sin = lambda pos, i: np.sin(pos / (1000 ** (i / num_units)))  # i ?????????
        cos = lambda pos, i: np.cos(pos / (1000 ** ((i - 1) / num_units)))  # i ?????????
        PE = [[sin(pos, i) if i % 2 == 0 else cos(pos, i) for i in range(num_units)] \
              for pos in range(MAX_WINDOW_SIZE)]

        PE = torch.tensor(np.array(PE), dtype=torch.float32).cuda()  # [MAX_WINDOW_SIZE, num_units]
        return PE

    def mask_fn(self):
        tmp = torch.unsqueeze(self.windows_utts_lens.cuda(), -1) #tmp: [batch_size, window_size, 1]
        self.mask = -torch.eq(tmp, 0.).float() * INF #mask: [batch_size, window_size, 1]
        #????????????????????????mask???????????????????????????nan
        self.position_encoding = self._position_encoding() #[MAX_WINDOW_SIZE, num_units]
        self.position_encoding = torch.unsqueeze(  
            torch.unsqueeze(
                self.position_encoding[:self.window_size],
                0
            ),
            2
        )
        self.q_status = self._attention(self.status_candidate_c, self.status_utt_h, self.status_utt_h) \
            + self.position_encoding #Matching Module, eq3 in paper, self.status_candidate_c: [slot_value_num, num_units]
        # [batch_size, window_size, status_num, num_units]

    def reinit(self):
        self.infos = dict()
        for dataset in ('train', 'dev', 'test'):
            self.infos[dataset] = dict()
            for slot in self.slots:
                self.infos[dataset][slot] = {
                    'ps': [],
                    'rs': [],
                    'f1s' : [],
                    'losses': []
                }
            self.infos[dataset]['global'] = {
                'ps': [],
                'rs': [],
                'f1s' : [],
                'losses': []
            }

    def _evaluate(self, pred_labels, gold_labels):
        def _add_ex_col(x):
            col = 1 - np.sum(x, -1).astype(np.bool).astype(np.float32)
            col = np.expand_dims(col, -1)
            x = np.concatenate([x, col], -1)
            return x
        pred_labels = _add_ex_col(pred_labels)
        gold_labels = _add_ex_col(gold_labels)
        tp = np.sum((pred_labels == gold_labels).astype(np.float32) * pred_labels, -1)
        pred_pos_num = np.sum(pred_labels, -1)
        gold_pos_num = np.sum(gold_labels, -1)
        p = (tp / pred_pos_num)
        r = (tp / gold_pos_num)
        p_add_r = p + r
        p_add_r = p_add_r + (p_add_r == 0).astype(np.float32)
        f1 = 2 * p * r / p_add_r

        return p, r, f1

    def evaluate(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        slots_pred_labels, slots_gold_labels = \
            self.inference(name, num, batch_size)

        info = dict()
        for slot in self.slots:
            info[slot] = {
                'p': None,
                'r': None,
                'f1': None
            }
        info['global'] = {
            'p': None,
            'r': None,
            'f1': None
        }

        for i, (slot_pred_labels, slot_gold_labels) in \
            enumerate(zip(slots_pred_labels, slots_gold_labels)):
            p, r, f1 = map(
                lambda x: float(np.mean(x)),
                self._evaluate(slot_pred_labels, slot_gold_labels)
            )
            slot = self.slots[i]
            info[slot]['p'] = p
            info[slot]['r'] = r
            info[slot]['f1'] = f1

        pred_labels = np.concatenate(slots_pred_labels, -1)
        gold_labels = np.concatenate(slots_gold_labels, -1)

        p, r, f1 = map(
            lambda x: float(np.mean(x)),
            self._evaluate(pred_labels, gold_labels)
        )
        info['global']['p'] = p
        info['global']['r'] = r
        info['global']['f1'] = f1

        return info
        
    def inference(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_pred_labels = [[] for i in range(len(self.slots_pred_labels))]
        slots_gold_labels = [[] for i in range(len(self.slots_pred_labels))]
        with torch.no_grad():
            for i, batch in enumerate(self.data.batch(name, batch_size, False)):
                if (i + 1) * batch_size > num:
                    break
                windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
                # print(windows_utts_batch.shape, windows_utts_lens_batch.shape, labels_batch.shape)

                self.set_input(windows_utts_batch, windows_utts_lens_batch, labels_batch)
                self._get_embedding()
                uinput = self.bert(self.utts).last_hidden_state
                for slot in self.Umodel_names:
                    cinput = self.bert(self.candidate_seqs_dict[slot]).last_hidden_state
                    utt_h, candidate_c = self.Umodels[slot](uinput, self.utts_lens, cinput, self.candidate_seqs_lens_dict[slot])
                    if slot == self.ontology.mutual_slot:
                        self.status_utt_h = utt_h
                        self.status_candidate_c = candidate_c
                    else:
                        self.slot_utt_hs_dict[slot] = utt_h
                        self.slot_candidate_cs_dict[slot] = candidate_c
                self.mask_fn()
                start = 0
                for i,slot in enumerate(self.Dmodel_names):
                    slot_pred_labels, _ , _ = self.Dmodels[slot](self.slot_utt_hs_dict[slot],
                                                            self.slot_candidate_cs_dict[slot],
                                                            self.status_candidate_c, self.position_encoding,
                                                            self.q_status, self.mask)

                    end = start + slot_pred_labels.shape[1]
                    slots_gold_labels[i].append(labels_batch[:, start: end])
                    slots_pred_labels[i].append(slot_pred_labels.detach().cpu().numpy())
                    start = end
        # slots_pred_labels?????????num_slots????????????????????????????????????[num, n * num_statues]
        for i in range(len(slots_gold_labels)):
            slots_gold_labels[i] = np.concatenate(slots_gold_labels[i], 0)
            slots_pred_labels[i] = np.concatenate(slots_pred_labels[i], 0)

        return slots_pred_labels, slots_gold_labels

    def compute_loss(self, name, num=-1, batch_size=DEFAULT_BATCH_SIZE):
        if num < 0:
            num = INF
        slots_loss = [[] for i in range(len(self.slots_loss))]
        with torch.no_grad():
            for i, batch in enumerate(self.data.batch(name, batch_size, False)):
                if (i + 1) * batch_size > num:
                    break
                windows_utts_batch, windows_utts_lens_batch, labels_batch = batch

                self.set_input(windows_utts_batch, windows_utts_lens_batch, labels_batch)
                self._get_embedding()

                uinput = self.bert(self.utts).last_hidden_state
                for slot in self.Umodel_names:
                    cinput = self.bert(self.candidate_seqs_dict[slot]).last_hidden_state
                    utt_h, candidate_c = self.Umodels[slot](uinput, self.utts_lens, cinput, self.candidate_seqs_lens_dict[slot])
                    if slot == self.ontology.mutual_slot:
                        self.status_utt_h = utt_h
                        self.status_candidate_c = candidate_c
                    else:
                        self.slot_utt_hs_dict[slot] = utt_h
                        self.slot_candidate_cs_dict[slot] = candidate_c
                self.mask_fn()
                start = 0
                slots_loss_batch = []
                for slot in self.Dmodel_names:
                    slot_pred_labels, num_logits, slot_pred_logits = self.Dmodels[slot](self.slot_utt_hs_dict[slot],
                                                            self.slot_candidate_cs_dict[slot],
                                                            self.status_candidate_c, self.position_encoding,
                                                            self.q_status, self.mask)

                    # print(slot_pred_labels.shape, self.labels.shape)
                    slot_gold_labels = torch.tensor(self.labels[:, start: start + num_logits]).float()
                    # ??????slot???loss
                    slot_loss = self.criterion1(slot_pred_logits.to('cpu'),
                                            slot_gold_labels.to('cpu'))  # [batch_size, status_num * slot_value_num]
                    slots_loss_batch.append(slot_loss.detach().numpy())
                    start += num_logits
                for i, slot_loss_batch in enumerate(slots_loss_batch):
                    slots_loss[i].append(slot_loss_batch)

        for i in range(len(slots_loss)):
            slots_loss[i] = np.concatenate(slots_loss[i], 0)

        losses = dict([(slot, None) for slot in self.slots])
        losses['global'] = None

        for i, slot_loss in enumerate(slots_loss):
            slot = self.slots[i]
            loss = float(np.mean(slot_loss))
            losses[slot] = loss

        losses['global'] = float(np.mean(np.concatenate(slots_loss, -1)))

        return losses

    def set_requires_grad(self, requires_grad=True):
        for slot in self.Umodel_names:
            for param in self.Umodels[slot].parameters():
                param.requires_grad = requires_grad
        for slot in self.Dmodel_names:
            for param in self.Dmodels[slot].parameters():
                param.requires_grad = requires_grad
                
    def save(self, location, save_graph=True):
        if not os.path.exists(location):
            os.makedirs(location)
        with open(os.path.join(location, 'params.json'), 'w', encoding='utf8') as f:
            json.dump(self.params, f, indent=4, ensure_ascii=False)
        with open(os.path.join(location, 'infos.json'), 'w', encoding='utf8') as f:
            json.dump(self.infos, f, indent=4, ensure_ascii=False)
        for item in self.Umodel_names:
            if save_graph:
                torch.save(self.Umodels[item], os.path.join(location, item+'_Umodel.pt'))
            else:
                torch.save(self.Umodels[item].state_dict(), os.path.join(location, item+'_Umodel.pt'))
        for item in self.Dmodel_names:
            if save_graph:
                torch.save(self.Dmodels[item], os.path.join(location, item+'_Dmodel.pt'))
            else:
                torch.save(self.Dmodels[item].state_dict(), os.path.join(location, item+'_Dmodel.pt'))

    def load(self,location):
        with open(os.path.join(location, 'params.json'), 'r', encoding='utf8') as f:
            self.params = json.load(f)
        with open(os.path.join(location, 'infos.json'), 'r', encoding='utf8') as f:
            self.infos = json.load(f)
        for item in self.Umodel_names:
            self.Umodels[item].load_state_dict(torch.load(os.path.join(location, item+'_Umodel.pt')))
        for item in self.Dmodel_names:
            self.Dmodels[item].load_state_dict(torch.load(os.path.join(location, item+'_Dmodel.pt')))
            
    def train(self,
            epoch_num,
            batch_size,
            tbatch_size,
            start_lr,
            end_lr,
            location=None):
        
        self.set_requires_grad(True)
        # ???????????????
        decay = (end_lr / start_lr) ** (1 / epoch_num)
        lr = start_lr
        self.slots_loss = [[] for i in self.Dmodel_names]
        for i in range(epoch_num):
            pbar = tqdm(
                self.data.batch('train', batch_size, True),
                desc='Epoch {}:'.format(i + 1),
                total=ceil(self.data.datasets['train']['num'] / batch_size)
            )
            self.set_train_mode()
            for batch in pbar:
                windows_utts_batch, windows_utts_lens_batch, labels_batch = batch
                self.set_window_size(windows_utts_batch)
                self.set_input(windows_utts_batch, windows_utts_lens_batch, labels_batch)
                self._get_embedding()
                uinput = self.bert(self.utts).last_hidden_state
                c_input = dict()
                for slot in self.Umodel_names:
                    c_input[slot] = self.bert(self.candidate_seqs_dict[slot]).last_hidden_state
                for slot in self.Umodel_names:
                    utt_h, candidate_c = self.Umodels[slot](uinput, self.utts_lens, c_input[slot], self.candidate_seqs_lens_dict[slot])
                    if slot == self.ontology.mutual_slot:
                        self.status_utt_h = utt_h
                        self.status_candidate_c = candidate_c
                    else:
                        self.slot_utt_hs_dict[slot] = utt_h
                        self.slot_candidate_cs_dict[slot] = candidate_c
                self.mask_fn()
                start = 0
                self.slots_pred_labels = []
                for l,slot in enumerate(self.Dmodel_names):
                    slot_pred_labels, num, slot_pred_logits = self.Dmodels[slot](self.slot_utt_hs_dict[slot], self.slot_candidate_cs_dict[slot],
                                       self.status_candidate_c, self.position_encoding, self.q_status, self.mask)
                    #print(slot_pred_labels.requires_grad)
                    self.slots_pred_labels.append(slot_pred_labels)
                    #print(slot_pred_labels.shape, self.labels.shape)
                    slot_gold_labels = torch.tensor(self.labels[:, start: start + num]).float()
                    # ??????slot???loss
                    p, r, f1 = map(lambda x: float(np.mean(x)),self._evaluate(slot_pred_labels.cpu().numpy(), slot_gold_labels.cpu().numpy()))
                    slot_loss = self.criterion(slot_pred_logits, slot_gold_labels.cuda())  # [batch_size, status_num * slot_value_num]
                    #print(slot_loss.requires_grad)
                    #slot_loss = torch.mean(slot_loss)
                    slot_loss.backward(retain_graph=True)
                    self.slots_loss[l].append(slot_loss.detach().to('cpu'))
                    # print(self.opt[slot].param_groups)
                    start += num
                for l,slot in enumerate(self.Dmodel_names):
                    self.opt[slot].step()
                    self.opt[slot].zero_grad()
                self.mutual_opt.step()
                self.mutual_opt.zero_grad()
            torch.cuda.empty_cache()
            pbar.close()
            lr *= decay
            self.set_eval_mode()
            train_prf = self.evaluate('train', tbatch_size, tbatch_size)
            torch.cuda.empty_cache()
            train_loss = self.compute_loss('train', tbatch_size, tbatch_size)
            torch.cuda.empty_cache()
            dev_prf = self.evaluate('dev', batch_size=tbatch_size)
            torch.cuda.empty_cache()
            dev_loss = self.compute_loss('dev', batch_size=tbatch_size)
            torch.cuda.empty_cache()
            test_prf = self.evaluate('test', batch_size=tbatch_size)
            test_loss = self.compute_loss('dev', batch_size=tbatch_size)
            torch.cuda.empty_cache()

            self._add_infos('train', train_prf)
            self._add_infos('train', train_loss)
            self._add_infos('dev', dev_prf)
            self._add_infos('dev', dev_loss)
            self._add_infos('test', test_prf)
            self._add_infos('test', test_loss)

            # ????????????
            print('Epoch {}: train_loss={:.4}, dev_loss={:.4}\
                train_p={:.4}, train_r={:.4}, train_f1={:.4}\
                dev_p={:.4}, dev_r={:.4}, dev_f1={:.4}'.
                format(i + 1, train_loss['global'], dev_loss['global'],
                    train_prf['global']['p'], train_prf['global']['r'],
                    train_prf['global']['f1'], dev_prf['global']['p'],
                    dev_prf['global']['r'], dev_prf['global']['f1']))
            
            if len(self.infos['dev']['global']['f1s']) > 0 and location:
                if dev_prf['global']['f1'] >= max(self.infos['test']['global']['f1s']):
                    save_graph = False
                    self.save(location, save_graph)
                    print('?????????{}???'.format(location))
            print('Now test result: f1={:.4}'.format(test_prf['global']['f1']))
            
    def _add_infos(self, name, info):
        for slot in info.keys():
            if isinstance(info[slot], float):
                # ?????????loss
                self.infos[name][slot]['losses'].append(info[slot])
            elif isinstance(info[slot], dict):
                # ?????????p r f
                for key in info[slot].keys():
                    self.infos[name][slot][key + 's'].append(info[slot][key])

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
                print('%s loaded' % name)

