#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
from utils import time_utc
import datetime as dt
import pandas as pd
from dateutil.parser import *
import pytz
import datetime
import talib as ta
import pickle
import torch
import numpy as np
import Feature_Function as FFF



import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import torch.nn.functional as F

from OandaAPI import OandaAPI


# In[15]:


class Encoder(nn.Module):
    def __init__(self,embedding_dim,enc_hidden_dim,dec_hidden_dim,dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.enc_hidden_dim = enc_hidden_dim 
        self.dec_hidden_dim = dec_hidden_dim 
        self.rnn = nn.LSTM(embedding_dim,enc_hidden_dim,bidirectional=True,batch_first=True) 
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,source_data):
        outputs,(hidden,cell_state) = self.rnn(source_data)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))) 
        return outputs,hidden 
    
class Attention(nn.Module):
    def __init__(self,enc_hidden_dim,dec_hidden_dim):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attn = nn.Linear((enc_hidden_dim*2) + dec_hidden_dim, dec_hidden_dim)


        self.vec = nn.Parameter(torch.rand(dec_hidden_dim))
        
    def forward(self,hidden,encoder_outputs): 

        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1,src_len,1)

        
        encoder_outputs = encoder_outputs.permute(0,1,2)
        association = torch.tanh(self.attn(torch.cat((hidden,encoder_outputs),dim=2)))
        association = association.permute(0,2,1)
        vec = self.vec.repeat(batch_size,1).unsqueeze(1)
        attention = torch.bmm(vec,association).squeeze(1)
        return F.softmax(attention,dim=1)
    
class Decoder(nn.Module):
    def __init__(self,embedding_dim,enc_hid_dim,dec_hid_dim, dropout,attention): 
        super().__init__()
        self.embedding_dim = embedding_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attention = attention
        self.rnn = nn.GRU((enc_hid_dim*2), dec_hid_dim,batch_first=True) 
        self.out = nn.Linear((enc_hid_dim*2)+ dec_hid_dim,1) 
        self.dropout = nn.Dropout(dropout)
    def forward(self,hidden,encoder_outputs):        
        a = self.attention(hidden,encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(0,1,2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(0,1,2)
        rnn_input = weighted
        output, hidden = self.rnn(rnn_input,hidden.unsqueeze(0))
        output2 = output.squeeze(1)
        weighted2 = weighted.squeeze(1)
        final_output = self.out(torch.cat((output2,weighted2),dim=1))
        return final_output, hidden.squeeze(0)
class Final_predictor(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self,source):
        encoder_outputs,hidden = self.encoder(source)
        output,hidden = self.decoder(hidden,encoder_outputs)
        
        return output
    
with open("k_means_rsi50.pkl","rb") as f:
    L_Kmeans = pickle.load(f)   
    
new_model = torch.load('attention_trained_rsi50.pt')
new_model.eval()
new_model2 = torch.load('attention_trained_rsi50_cluster7.pt')
new_model2.eval()   
    
    
training_features1=['mid_o','mid_c','PLUS_DM','ADD','MULT','ROC']
training_features2=['mid_o','mid_c','PLUS_DM','ADD','MULT','CDLCONCEALBABYSWALL']
    
no_candle=0
no_trade=0
api = OandaAPI()    
confidence = 0.0003
confidence2 = 0.0003
    
SLEEP=10.0
latest_time = None
if __name__ == '__main__':
    while True:
        ################
        ##check open position
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        print('-------------------------------------------------')
        now = datetime.datetime.now(tz=pytz.UTC)
        print('Checking Open Postion...')
        try:
            open_pos,status = api.open_trades()
            print('Open position total number: ',len(open_pos))
            for x in open_pos:
                if now - x.openTime >= dt.timedelta(minutes=29):
                    sta = api.close_trade(x.trade_id)
                    print('Trade number [{}] closed: {}'.format(x.trade_id,sta))
            open_pos2,status2 = api.open_trades() 
            print('Open position total number after closure (if any): ',len(open_pos2))
        except:
            continue
        
        print('-------------------------------')
        print('--------------------------------')
        print('                                   ')
        
        ################

        a,df = api.fetch_candles('AUD_USD', count=30, granularity="M15")
        try:
            c = df['time'].iloc[-1]

            import pytz
            import datetime
            now = datetime.datetime.now(tz=pytz.UTC)
            ooo = c.to_pydatetime()
            print('current time: ',now)
            print('candle_time: ',ooo)
            if now-ooo <= dt.timedelta(minutes=3) and latest_time != ooo:
                no_candle+=1
        #         print('True')
        #     else:
        #         print("False")
                df['rsi'] = ta.RSI(df['mid_c'],14)
                df['PLUS_DM'] = ta.PLUS_DM(df['mid_h'],df['mid_l'],timeperiod=14)
                df['ADD'] = ta.ADD(df['mid_h'],df['mid_l'])
                df['MULT'] = ta.MULT(df['mid_h'],df['mid_l'])
                df['ROC'] = ta.ROC(df['mid_c'],timeperiod=10)
                df['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(df['mid_o'],df['mid_h'],df['mid_l'],df['mid_c'])

                print('current time: ', now)
                print('Candle # {} at time:{}'.format(no_candle,ooo))
                print('--------------------------------------------')
                df1 = df.iloc[-10:]
                print(df1)
                pr = df1.iloc[-1]['mid_c']+confidence
                pr2 = df1.iloc[-1]['mid_c']-confidence
                pr3 = df1.iloc[-1]['mid_c']+confidence2+0.0001
                pr4 = df1.iloc[-1]['mid_c']-confidence2

                if df1.iloc[-1]['rsi'] <=50:

                    feature = FFF.myfeature(training_features1,9,9,0,0,df1)
                    feature_con = FFF.myfeatureConventional(training_features2,9,9,0,0,df1)
                    cluster= L_Kmeans.predict(np.array(feature_con).reshape(1,-1))
                    if cluster==1:
                        with torch.no_grad():
                            pred = new_model(torch.tensor(feature,dtype=torch.float32).unsqueeze(0))
                            preds = pred.item()
                            if preds >= df1.iloc[-1]['mid_c']:#+confidence:# and preds < df1.iloc[-1]['mid_c']+confidence:
                                try:
                                    id,ok = api.place_trade('AUD_USD',units=500000,take_profit=round(pr,5),stop_loss=round(pr2,5))
                                    no_trade+=1
                                except:
                                    id = 'ERROR with placing order'
#                             elif preds >=df1.iloc[-1]['mid_c']+confidence:
#                                 try:
#                                     id,ok = api.place_trade('AUD_USD',units=720000,take_profit=round(preds,5))
#                                     no_trade+=1
#                                 except:
#                                     id = 'ERROR with placing order'
                                print('!!!!!!!Cluster [{}]; TRADE # [{}] with id: [{}], status [{}], WITH PREDICTED PRICE: {}                                                     '.format(cluster,no_trade,id,ok,preds))
                            else:
                                print('no trade as predicted price is too close to entry close.')
            
                        print('-----------------------------------------------------------------')
                    elif cluster==7:
                        with torch.no_grad():
                            pred = new_model2(torch.tensor(feature,dtype=torch.float32).unsqueeze(0))
                            preds = pred.item()
                            if preds > df1.iloc[-1]['mid_c']+confidence2*2:#+confidence:# and preds < df1.iloc[-1]['mid_c']+confidence:
                                try:
                                    id,ok = api.place_trade('AUD_USD',units=500000,take_profit=round(pr3,5),stop_loss=round(pr4,5))
                                    no_trade+=1
                                except:
                                    id = 'ERROR with placing order'
#                             elif preds >=df1.iloc[-1]['mid_c']+confidence:
#                                 try:
#                                     id,ok = api.place_trade('AUD_USD',units=720000,take_profit=round(preds,5))
#                                     no_trade+=1
#                                 except:
#                                     id = 'ERROR with placing order'
                                print('!!!!!!!!!!!!!!Cluster [{}] TRADE # [{}] with id: [{}], status [{}], WITH PREDICTED PRICE: {}                                                     '.format(cluster,no_trade,id,ok,preds))
                            else:
                                print('no trade as predicted price is too close to entry close.')
            
                        print('-----------------------------------------------------------------')
                    else:
                        print('?????No trade wtih cluster number: ', cluster)
                        print('-----------------------------------------------------------------')


                latest_time = ooo
                print('latest candle time: ',latest_time)

            else:
                print('no update yet for current time: ', now)
                
        except:
            print(f'something went wrong at {datetime.datetime.now(tz=pytz.UTC)}, so skip to next candle')
            
        

        print('Sleeping...')
        import time

        time_left = 50
        while time_left > 0:
            print('Time until next refresh (in seconds):',time_left)
            time.sleep(5)
            time_left = time_left - 5


# In[ ]:




