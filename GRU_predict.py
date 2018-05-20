import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import datetime

def run(hidden_dim, dropout, batch_size):
    # hyper parameter
    train_time = None
    final_loss = None
    INPUT_SIZE = 1
    HIDDEN_DIM = hidden_dim
    BATCH_SIZE = batch_size
    DROUPOUT = dropout
    TIME_STEP = 10

    # data import
    f = open('Lorenz.csv')
    data = pd.read_csv(f)

    data = data.iloc[:,0].values
    nor = sum(data)/len(list(data))
    data = data/nor
    # nor = sum(data)/data
    train_data = data[0:2000]
    test_data = data[2000:2500]

    def trainDatGen(seq,k):
        in_dat = list()
        out_dat = list()
        L = len(seq)
        for i in range(L - k - 5):
            indat = seq[i:i + k]
            outdat = seq[i + 5:i + k + 5]
            in_dat.append(indat)
            out_dat.append(outdat)
        return in_dat, out_dat

    def testDatGen(seq,k):
        dat = list()
        L = len(seq)
        for i in range(L - k - 5):
            indat = seq[i:i + k]
            outdat = seq[i + 5:i + k + 5]
            dat.append((indat, outdat))
        return dat

    def ToVariable(x):
        tmp = torch.FloatTensor(x)
        return Variable(tmp)


    in_dat,out_dat = trainDatGen(train_data,10)
    in_dat = torch.FloatTensor(in_dat)
    out_dat = torch.FloatTensor(out_dat)

    torch_dataset = Data.TensorDataset(data_tensor=in_dat,target_tensor=out_dat)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False)

    class LSTMpred(nn.Module):
        def __init__(self, input_size, hidden_dim, batch_size, dropout):
            super(LSTMpred, self).__init__()
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.batch_size = batch_size
            self.dropout = dropout
            self.lstm = nn.GRU(
                dropout=self.dropout,
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                batch_first=True,
                num_layers=1)
            self.hidden2out = nn.Linear(hidden_dim, 1)
            self.hidden = None

        def init_hidden(self):
            # (num_layer, batch_size, hidden_dim)
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

        def init_hidden_1(self):
            # (num_layer, batch_size, hidden_dim)
            return (Variable(torch.zeros(1, 1, self.hidden_dim)))

        def forward(self, seq):
            lstm_out, self.hidden = self.lstm(
                seq, self.hidden)
            outdat = self.hidden2out(lstm_out)
            return outdat

    model = LSTMpred(INPUT_SIZE, HIDDEN_DIM, BATCH_SIZE, DROUPOUT)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = None
    loss_seq = []
    startTime = datetime.datetime.now()
    for epoch in range(10):
        # print('epoch:',epoch,'|')
        for step,(batch_x,batch_y) in enumerate(loader):
            train_x = Variable(batch_x)
            train_y = Variable(batch_y)
            if(len(train_x) == BATCH_SIZE):
                # (batch_size, time_step, input_size)
                train_x = train_x.view(BATCH_SIZE, -1, INPUT_SIZE)
                train_y = train_y.view(BATCH_SIZE, -1, INPUT_SIZE)

                optimizer.zero_grad()
                model.hidden = model.init_hidden()
                train_out = model(train_x)
                loss = loss_function(train_out,train_y)
                loss.backward()
                optimizer.step()
            else:
                pass

        # if epoch == 9:
        #     print(time)
        # print(loss)
        loss_seq.append(loss.data.numpy())
        final_loss = loss
    endTime = datetime.datetime.now()
    time = endTime - startTime
    train_time = time

    test_data = testDatGen(test_data,10)

    predDat = []
    trueDat = []

    model.hidden = model.init_hidden_1()
    for test_in, trueVal in test_data:
        test_in = ToVariable(test_in)
        trueVal = ToVariable(trueVal)
        # (batch_size,time_step,input_size)
        test_in = test_in.view(1, 10, 1)
        test_out = model(test_in)
        test_out = test_out.view(10)
        predDat.append(test_out[-1].data.numpy())
        trueVal = trueVal.view(10)
        trueDat.append(trueVal[-1].data.numpy())

    # plt.figure()
    # plt.plot(trueDat,color='red')
    # plt.plot(predDat,color='blue')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(loss_seq,color='blue')
    # plt.show()

    return train_time,final_loss