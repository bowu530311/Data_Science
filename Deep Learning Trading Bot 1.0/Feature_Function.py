#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def myfeatureConventional(feature_list,data_position,timestepback,timestepforward,future_outlook_window,data):
    # Future_outlook_window is set at minimium value of 1
    # timestepforward can be set as 0, this will impact the feature sequence/length
    # timestepback will decide n sessions of data that we used in feature in order to make predictions
    final=[]
    time_step_feature=[]
    for x in range(data_position-timestepback,data_position+timestepforward+1):
        for aa in feature_list:
            time_step_feature.append(data.iloc[x][aa])
    final.append(time_step_feature)
    
#     target = df1.iloc[data_position+timestepforward+future_outlook_window]['close']
    
#     mx=[]
#     mn=[]
#     for bb in range(future_outlook_window):
#         mx.append(df1.iloc[data_position+timestepforward+bb+1]['high'])
#         mn.append(df1.iloc[data_position+timestepforward+bb+1]['low'])
#     max_price=max(mx)
#     min_price=min(mn)
    
    
    
    
    
    return time_step_feature#,target,max_price,min_price



def myfeature(feature_list,data_position,timestepback,timestepforward,future_outlook_window,data):
    # Future_outlook_window is set at minimium value of 1
    # timestepforward can be set as 0, this will impact the feature sequence/length
    # timestepback will decide n sessions of data that we used in feature in order to make predictions
    final=[]
    time_step_feature=[]
    for x in range(data_position-timestepback,data_position+timestepforward+1):
        feature=[]
        for aa in feature_list:
            feature.append(data.iloc[x][aa])
        time_step_feature.append(feature)
    final.append(time_step_feature)
    #final = torch.tensor(final).unsqueeze(0)
    
#     target = df1.iloc[data_position+timestepforward+future_outlook_window]['close']
    
#     mx=[]
#     mn=[]
#     for bb in range(future_outlook_window):
#         mx.append(df1.iloc[data_position+timestepforward+bb+1]['high'])
#         mn.append(df1.iloc[data_position+timestepforward+bb+1]['low'])
#     max_price=max(mx)
#     min_price=min(mn)
    
    
    
    
    
    return time_step_feature#,target,max_price,min_price

