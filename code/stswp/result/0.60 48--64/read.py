import numpy as np
import matplotlib.pyplot as plt  


loss = []
pearson = []
with open('val.log','r') as f:
    for i in f:
        # print(i.split('\t')[1].split('(')[1].split(',')[0])
        # exit()
        pearson.append(float(i.split('\t')[2]))
        loss.append(float(i.split('\t')[1].split('(')[1].split(',')[0])*25)
# print(pearson)
# print(loss)
# exit()

x=np.arange(105)


l1=plt.plot(x[:22],pearson[:22],'-',label='Pearson')
# l1=plt.plot(x[:22],loss[:22],'-',label='Loss')


# l2=plt.plot(x2,y2,'g--',label='type2')
# l3=plt.plot(x3,y3,'b--',label='type3')
# plt.plot(x,loss,'b.')#,x2,y2,'g+-',x3,y3,'b^-')
plt.title('Validation Evaluation')
plt.xlabel('Epoch')
plt.ylabel('Evaluation')
plt.legend()
plt.show()

