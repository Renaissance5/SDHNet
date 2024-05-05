import idna
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.decompose import series_decomp
from layers.DualBlock import DoubleSample,ShortExtractor,LongExtractor,FeatureUnion

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.dropout = nn.Dropout(p=configs.dropout)
        self.pred_len=configs.pred_len
        self.seq_len=configs.seq_len
        self.period=configs.period
        self.nums=int(self.seq_len/self.period)
        if configs.features=='M':
            self.channels=configs.enc_in
        elif configs.features=='S':
            self.channels=1
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}
        self.activation=self.activations[configs.activation]

        # decomposition
        self.decompsition = series_decomp(configs.moving_avg)

        # trend-cyclical prediction
        self.trend_project=nn.Linear(in_features=configs.seq_len,out_features=configs.pred_len)

        # seasonal prediction
        self.sample=DoubleSample(self.nums)
        self.shortExtractor=ShortExtractor(configs.pooling_size,configs.seq_len,self.nums,
                                           configs.d_model,configs.dropout,configs.kernel_list,self.activation)
        self.longExtractor=LongExtractor(self.nums,configs.d_model)
        self.season_project=nn.Linear(in_features=configs.d_model,out_features=configs.pred_len) 

    def forward(self,batch_x,batch_y):

        
        seq_last = batch_x[:,-1:,:].detach()
        x = batch_x - seq_last
        
        # batch_x:(batch, seq_len, channel)
        season_init,trend_init=self.decompsition(x)
        # (b,s,c)->(b,c,s)->(bc,s)
        season_init=season_init.permute(0,2,1).reshape(-1,self.seq_len)
        # (b,s,c)->(b,c,s)->(bc,s)
        trend_init=trend_init.permute(0,2,1).reshape(-1,self.seq_len)  
        # (bc,s)->(bc,pred_len)
        trend_out=self.trend_project(trend_init)
        #(bc,pred_len)->(b,c,pred_len)->(b,pred_len,c)                                  
        trend_out=trend_out.reshape(-1,self.channels,self.pred_len).permute(0,2,1)

        con_x,eq_x=self.sample(season_init)
        tokens=self.shortExtractor(con_x)
        h_t=self.longExtractor(eq_x)

        out=FeatureUnion(h_t,tokens)
        out=self.season_project(out)
        #(bc,pred_len)->(b,c,pred_len)->(b,pred_len,c)
        season_out=out.reshape(-1,self.channels,self.pred_len).permute(0,2,1)

        res=trend_out+season_out
        res=res+seq_last

        return res