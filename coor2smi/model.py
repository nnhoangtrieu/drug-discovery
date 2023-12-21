import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn 
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN_Attention(nn.Module): # Neural Network Attention 
    def __init__(self, dim_model):
        super(NN_Attention, self).__init__()
        self.Wa = nn.Linear(dim_model, dim_model)
        self.Ua = nn.Linear(dim_model, dim_model)
        self.Va = nn.Linear(dim_model, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)

        context = torch.bmm(weights, keys)

        return context, weights 
    

class DP_Attention(nn.Module) : # Dot Product Attention
    def __init__(self, dim_model, num_head) :
        super(DP_Attention, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_head = dim_model // num_head

        self.Q = nn.Linear(dim_model, dim_model)
        self.K = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model, dim_model)

        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, Q, K, V) :
        B = Q.size(0) # Shape Q, K, V: (B, longest_smi, dim_model)

        Q, K, V = self.Q(Q), self.K(K), self.V(V)

        len_Q, len_K, len_V = Q.size(1), K.size(1), V.size(1)

        Q = Q.reshape(B, self.num_head, len_Q, self.dim_head)
        K = K.reshape(B, self.num_head, len_K, self.dim_head)
        V = V.reshape(B, self.num_head, len_V, self.dim_head)
        
        K_T = K.transpose(2,3).contiguous()

        attn_score = Q @ K_T

        attn_score = attn_score / (self.dim_head ** 1/2)

        attn_distribution = torch.softmax(attn_score, dim = -1)

        attn = attn_distribution @ V

        attn = attn.reshape(B, len_Q, self.num_head * self.dim_head)
        
        attn = self.out(attn)

        return attn, attn_distribution
    

class Encoder(nn.Module) :
    def __init__(self, dim_model, num_head, dropout) :
        super(Encoder, self).__init__()
        self.Self_Attention = DP_Attention(dim_model, num_head) 
        self.LSTM = nn.LSTM(2 * dim_model, dim_model, batch_first=True)
        self.Up_Size = nn.Linear(3, dim_model)
        self.Dropout = nn.Dropout(dropout)
    
    def forward(self, x) :
        x = self.Dropout(self.Up_Size(x))

        attn, self_attn = self.Self_Attention(x, x, x) 

        input_lstm = torch.cat((attn, x), dim = -1)

        e_all, (h, c) = self.LSTM(input_lstm)

        return e_all, h, c, self_attn
    
class Decoder(nn.Module) :
    def __init__(self, dim_model, num_head, output_size, longest_smi, dropout) :
        super(Decoder, self).__init__()
        self.longest_smi = longest_smi
        self.Embedding = nn.Embedding(longest_smi, dim_model)
        self.Cross_Attention = NN_Attention(dim_model) 
        self.Dropout = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(2 * dim_model, dim_model, batch_first=True)
        self.Linear = nn.Linear(dim_model, output_size)
    def forward(self, e_all, e_h, e_c, target = None) :
        B = e_all.size(0)

        d_input = torch.zeros(B, 1, dtype=torch.long, device = device)

        d_h, d_c = e_h, e_c 

        outputs, cross_attn = [], [] 

        for i in range(self.longest_smi) : 
            output, d_h, d_c, step_attn = self.forward_step(d_input, d_h, d_c, e_all)

            outputs.append(output), cross_attn.append(step_attn)

            if target is not None :
                d_input = target[:, i].unsqueeze(1)
            else : 
                _, topi = output.topk(1)
                d_input = topi.squeeze(-1).detach()

        
        outputs = torch.cat(outputs, dim = 1)
        outputs = F.log_softmax(outputs, dim = -1) 

        cross_attn = torch.cat(cross_attn, dim = 1)

        return outputs, cross_attn

    def forward_step(self, d_input, d_h, d_c, e_all) :
        embedded = self.Dropout(self.Embedding(d_input))
        
        query = d_h.permute(1, 0, 2) + d_c.permute(1, 0, 2)

        attn, cross_attn = self.Cross_Attention(query, e_all)

        input_gru = torch.cat((embedded, attn), dim = 2)

        output, (d_h, d_c) = self.LSTM(input_gru, (d_h, d_c)) 

        output = self.Linear(output) 

        return output, d_h, d_c, cross_attn