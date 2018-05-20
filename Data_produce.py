import GRU_predict
import numpy as np
import matplotlib.pyplot as plt
# GRU_predict.run(hidden_dim, dropout, batch_size)
hidden_dim = 30
for i in range(0,4):
    i = 30 + i*20
    train_time,final_loss = GRU_predict.run(hidden_dim,0.9,i)
    print(train_time)
    print(final_loss)

