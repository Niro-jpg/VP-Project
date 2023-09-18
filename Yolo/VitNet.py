
import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = self.emb_dim // num_heads

        # Inizializzazione dei moduli lineari per proiettare Q, K, V e l'output.
        self.W_q = nn.Linear(self.emb_dim, self.emb_dim)
        self.W_k = nn.Linear(self.emb_dim, self.emb_dim)
        self.W_v = nn.Linear(self.emb_dim, self.emb_dim)
        self.W_o = nn.Linear(self.emb_dim, self.emb_dim)


    def forward(self, q,k,v):
        q = self.split_heads(self.W_q(q))  #64,2,22,2
        k = self.split_heads(self.W_k(k))  #64,2,2,22
        v = self.split_heads(self.W_v(v))


        att = self.att_score(q, k, v)  #62,2,22,22
        out = self.W_o(self.combine_heads(att))

        return out

    def att_score(self, q, k, v):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_dim)






class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, feedforward_dim=32, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multihead_attention = MultiHeadAttention(emb_dim, num_heads)

        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, emb_dim)
        )

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        att = self.multihead_attention(x,x,x)

        add_nor = self.layer_norm1(x + self.dropout(att))
        ff_out = self.feedforward(add_nor)
        out = self.layer_norm2(add_nor + self.dropout(ff_out))


        return out





class Encoder(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers=6, feedforward_dim=32, dropout=0.1):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(emb_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self, C, B = 2, S = 7, image_flat_dim = 9075, d_model=1024, num_heads=2, num_layers=6, d_ff=256, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model=d_model

        self.num_heads=num_heads

        self.transformer_encoder = Encoder(self.d_model,self.num_heads

        )

        self.fc = nn.Linear(d_model, (5 * B + C))
        self.miaos = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, S * S * (5 * B + C)),
        )
        self.embedding = nn.Linear(image_flat_dim, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src):

        batch_size = src.shape[0]
        imag_dim = src.shape[2]
        
        print(src.shape)
        # Dividi l'immagine in blocchi
        src = src.unfold(2, 22, 22).unfold(3, 22, 22)

        # Appiattisci i blocchi
        src = src.contiguous().view(batch_size, 64, -1)
        src = self.embedding(src) + self.positional_encoding(64 ,self.d_model)
        enc_output = self.transformer_encoder(src)
        output = self.fc(enc_output)
        
        print(output.shape)

        return output



    def positional_encoding(self, max_seq_length, d_model):

        positional_encoding = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, 1::2] = torch.cos(pos * div_term)
        return positional_encoding

