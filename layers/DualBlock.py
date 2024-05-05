import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleSample(nn.Module):
    def __init__(self,nums):
        super(DoubleSample,self).__init__()
        self.nums=nums
        
    def equidistant_samp(self,x):
        eq_x=x.view(x.size(0),self.nums,int(x.size(1)/self.nums))
        eq_x=eq_x.transpose(1,2)
        return eq_x               

    def continuous_samp(self,x):
        con_x=x.view(x.size(0),self.nums,int(x.size(1)/self.nums))
        return con_x              
    
    def forward(self,x):
        con_x=self.continuous_samp(x)
        eq_x=self.equidistant_samp(x)
        return con_x,eq_x
    
class MLP_short(nn.Module):
    def __init__(self,seq_len,nums,d_model=512,pooling_size=8,dropout=0.1):
        super(MLP_short,self).__init__()
        self.d_model=d_model
        self.dropout=nn.Dropout(p=dropout)
        self.sub_len=int(seq_len/nums)
        
        linear1=[]
        linear1+=[
            nn.Linear(in_features=int(nums+self.sub_len),out_features=nums),
            nn.Tanh(),
            self.dropout,
        ]
        self.linear1=nn.Sequential(*linear1)

        linear2=[]
        linear2+=[
            nn.Linear(in_features=int(self.d_model/pooling_size),out_features=self.d_model),
            self.dropout,
        ]
        self.linear2=nn.Sequential(*linear2)

    def forward(self,x):
        out=self.linear1(x.permute(0,2,1)).transpose(1,2)
        out=self.linear2(out)
        return out
    

class ShortExtractor(nn.Module):

    def __init__(self,pooling_size,seq_len,nums,d_model,dropout,kernel_list=[3,3],activation=nn.Tanh()):
        super(ShortExtractor,self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        self.activation=activation
        self.seq_len=seq_len
        self.nums=nums
        self.sub_len=int(seq_len/nums)
        self.k1=kernel_list[0]
        self.k2=kernel_list[1]
        self.pooling_mode=0
        self.pooling_size=pooling_size
        self.d_model=d_model

        self.temporal_conv1d=nn.Conv1d(in_channels=self.sub_len,out_channels=d_model,
                                       kernel_size=self.k1,stride=1,padding=self.k1//2,padding_mode='circular')
        nn.init.kaiming_normal_(self.temporal_conv1d.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.channel_conv1d=nn.Conv1d(in_channels=self.nums,out_channels=d_model,
                                      kernel_size=self.k2,stride=1,padding=self.k2//2,padding_mode='circular')
        nn.init.kaiming_normal_(self.channel_conv1d.weight, mode='fan_in', nonlinearity='leaky_relu')

        if self.pooling_mode==0:
            self.pooling=nn.AvgPool1d(kernel_size=self.pooling_size,stride=self.pooling_size)
        else:
            self.pooling=nn.MaxPool1d(kernel_size=self.pooling_size,stride=self.pooling_size)

        self.mlp=MLP_short(self.seq_len,self.nums,d_model,self.pooling_size,dropout)


    def forward(self,x):
        # temporal conv1d
        t_x=self.temporal_conv1d(x.permute(0,2,1)).transpose(1,2)
        t_x=self.activation(t_x)
        t_x=self.dropout(t_x)
        # temporal pooling
        t_x=self.pooling(t_x)

        # channel conv1d
        c_x=self.channel_conv1d(x)
        c_x=self.activation(c_x)
        c_x=self.dropout(c_x)
        # channel pooling
        c_x=self.pooling(c_x.permute(0,2,1)).transpose(1,2)

        # permute & concate
        out=torch.cat((t_x,c_x.permute(0,2,1)),dim=1)
        # mlp
        out=self.mlp(out)

        return out
    

class Single_GRU(nn.Module):

    def __init__(self,input_size,d_model):
        super(Single_GRU,self).__init__()

        self.GRU=nn.GRU(input_size=input_size,hidden_size=d_model,
                        num_layers=1,bias=True,batch_first=True)
        
    def forward(self,x):
        _,h_t=self.GRU(x)
        h_t=h_t.permute(1,0,2)
        return h_t

class Grouped_GRU(nn.Module):

    def __init__(self,input_size,d_model,group=3):
        super(Grouped_GRU,self).__init__()
        self.group=group
        self.GRU_layers=nn.ModuleList()

        for _ in range(self.group):
            self.GRU_layers.append(Single_GRU(input_size=input_size,d_model=d_model))

        self.GRU_union=Single_GRU(input_size=d_model,d_model=d_model)

    def forward(self,x):

        group_len=int(x.size(1)//self.group)
        out=[]
        for i in range(self.group):
            h_n=self.GRU_layers[i](x[:,group_len*i:group_len*(i+1),:])
            out.append(h_n)
        final_h_n=self.GRU_union(torch.cat(out,dim=1))
        return final_h_n

class LongExtractor(nn.Module):

    def __init__(self,input_size,d_model):
        super(LongExtractor,self).__init__()

        self.group_GRU=Grouped_GRU(input_size=input_size,d_model=d_model)
        
    def forward(self,x):
        h_n=self.group_GRU(x)
        return h_n

def FeatureUnion(h_n,token):
    X = torch.cat((token, h_n), dim=1)
    h_t_expend = h_n.expand(-1, token.size(1) + 1, -1)
    cos_sim = F.cosine_similarity(h_t_expend, X, dim=2)
    weights = F.softmax(cos_sim, dim=1)
    weighted_h = torch.sum(weights.unsqueeze(2) * X, dim=1)
    return weighted_h