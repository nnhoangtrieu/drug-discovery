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
    

class EncoderBlock(nn.Module) :
    def __init__(self, dim_model, num_head, fe, dropout) :
        super(EncoderBlock, self).__init__()

        self.Self_Attention = DP_Attention(dim_model,num_head)
        self.LSTM = nn.LSTM(input_size=2 * dim_model, hidden_size=dim_model, batch_first=True)

    def forward(self, Q, K, V) :

        attn, self_attn = self.Self_Attention(Q, Q, Q)

        input_lstm = torch.cat((Q, attn), dim = -1)

        all_state, (last_state, _) = self.LSTM(input_lstm)

        return all_state, last_state, self_attn


class Encoder(nn.Module) :
    def __init__(self, dim_model, num_block, num_head, len_dic, fe = 1, dropout = 0.1) :
        super(Encoder, self).__init__()

        self.Embedding = nn.Embedding(len_dic, dim_model)
        self.Dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(dim_model, num_head, fe, dropout) for _ in range(num_block)
        )

    def forward(self, x) :
        out = self.Dropout(self.Embedding(x))
        for block in self.encoder_blocks : 
            out, last_state, self_attn = block(out, out, out) 
        return out, last_state, self_attn
        

class GRU(nn.Module) :
    def __init__(self, dim_model, longest_coor,dropout, num_head = 1, output_size = 3) :
        super(GRU, self).__init__()

        self.longest_coor = longest_coor

        self.cross_attn_nn = NN_Attention(dim_model)

        self.gru = nn.GRU(3 + dim_model, dim_model, batch_first=True)

        self.out = nn.Linear(dim_model, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, e_all, e_last, target = None) :
        B = e_all.size(0)

        d_input = torch.zeros(B, 1, 3).to(device)

        d_hidden = e_last

        d_outputs, cross_attn = [], []

        for i in range(self.longest_coor) :
            d_output, d_hidden, step_attn = self.forward_step(d_input, d_hidden, e_all)

            d_outputs.append(d_output), cross_attn.append(step_attn)

            if target is not None :
                d_input = target[:, i, :].unsqueeze(1)
            else :
                d_input = d_output

        d_outputs = torch.cat(d_outputs, dim = 1)

        cross_attn = torch.cat(cross_attn, dim = 1)
        
        return d_outputs, d_hidden, cross_attn


    def forward_step(self, d_input, d_hidden, e_all) :
        Q = d_hidden.permute(1,0,2)

        d_input = self.dropout(d_input)

        attn, attn_distribution = self.cross_attn_nn(Q, e_all)

        input_lstm = torch.cat((attn, d_input), dim = 2)

        output, d_hidden = self.gru(input_lstm, d_hidden) 
        
        output = self.out(output)

        return output, d_hidden, attn_distribution
    

class DecoderBlock(nn.Module) :
    def __init__(self, dim_model, num_head, longest_coor, fe, dropout) :
        super(DecoderBlock, self).__init__()

        self.gru = GRU(dim_model, longest_coor,dropout, num_head)


    def forward(self, e_all, e_last, target = None) :
        output, _, cross_attn = self.gru(e_all, e_last, target)
        
        return output, cross_attn
    
class Decoder(nn.Module) :
    def __init__(self, dim_model,num_block, num_head, longest_coor, fe = 1, dropout = 0.1) :
        super(Decoder, self).__init__()

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(dim_model, num_head,longest_coor, fe, dropout) for _ in range(num_block)]
        )

        
    def forward(self, e_all, e_last, target = None) :
        for block in self.decoder_blocks :
            target, cross_attn = block(e_all, e_last, target)
        return target, cross_attn

