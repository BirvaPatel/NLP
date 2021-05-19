"""
Prepared By: BIRVA PATEL(1111092)
Subject: Non Linear Regression to implement a one dimentional 
		 convolution based NN for predicting median house value

"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.nn import Conv1d
from torch.nn import MaxPool1d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn.functional import relu
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import SGD
from torch.nn import L1Loss
from ignite.contrib.metrics.regression.r2_score import R2Score

# preprocessing of data
dataset=pd.read_csv('housing.csv')
dataset=dataset.dropna()
print("Here is the first ten row of the data set")
print(dataset.head(10))
x = dataset.iloc[0:20]

df = pd.DataFrame(x, columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                                'population','households','median_income'])

print(df.plot())
Y=dataset['median_house_value']
X=dataset.loc[:,'longitude':'median_income']

x_train,x_test,y_train,y_test = train_test_split(X, Y ,test_size=0.2)
x_train_np= x_train.to_numpy()
y_train_np= y_train.to_numpy()

x_test_np= x_test.to_numpy()
y_test_np= y_test.to_numpy()

# Regression model
class CnnRegressor(torch.nn.Module):
    def __init__(self,batch_size, inputs, outputs):
        super(CnnRegressor,self).__init__()
        self.batch_size=batch_size
        self.inputs=inputs
        self.outputs=outputs
        
        self.input_layer=Conv1d(inputs,batch_size,1)
        self.max_pooling_layer= MaxPool1d(1)
        self.conv_layer=Conv1d(batch_size,128,1)
        self.flatten_layer=Flatten()
        self.linear_layer=Linear(128,64)
        self.output_layer=Linear(64,outputs)
    def feed(self,input):
        input=input.reshape(self.batch_size,self.inputs,1)
        output=relu(self.input_layer(input))
        output=self.max_pooling_layer(output)
        output=relu(self.conv_layer(output))
        output=self.flatten_layer(output)
        output=self.linear_layer(output)
        output=self.output_layer(output)
        return output


# calling the regression model
batch_size=64
model=CnnRegressor(batch_size, X.shape[1], 1)
model.cuda()

# calculating the model loss
def model_loss(model,dataset,train=False,optimizer=None):
    performance=L1Loss()
    score_metric=R2Score()
    
    avg_loss=0
    avg_score=0
    count=0
    for input,output in iter(dataset):
        predictions=model.feed(input)
        loss=performance(predictions,output)
        score_metric.update([predictions,output])
        score=score_metric.compute()
        if(train):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss += loss.item()
        avg_score += score
        count += 1
    return avg_loss/count,avg_score/count

epochs = 50
optimizer = SGD(model.parameters(),lr=1e-5)
inputs=torch.from_numpy(x_train_np).cuda().float()
outputs=torch.from_numpy(y_train_np.reshape(y_train_np.shape[0],1)).cuda().float()
tensor= TensorDataset(inputs,outputs)
loader= DataLoader(tensor,batch_size,shuffle=True, drop_last=True)

#printing the loss and r2 score for each epoch
for epoch in range(epochs):
  avg_loss,avg_r2_score=model_loss(model,loader,train=True,optimizer=optimizer)
  print("Epoch"+str(epoch+1)+":\n\tLoss="+str(avg_loss)+"\n\tR^2Score="+str(avg_r2_score))

  
inputs = torch.from_numpy(x_test_np).cuda().float()
outputs = torch.from_numpy(y_test_np.reshape(y_test_np.shape[0], 1)).cuda().float()
tensor = TensorDataset(inputs,outputs)
loader = DataLoader(tensor, batch_size, shuffle=True,drop_last=True)

#printing the final average loss and r2score
avg_loss, avg_r2_score = model_loss(model, loader)
print("The model L1 loss is:" + str(avg_loss))
print("The model R^2 loss is:" + str(avg_r2_score))