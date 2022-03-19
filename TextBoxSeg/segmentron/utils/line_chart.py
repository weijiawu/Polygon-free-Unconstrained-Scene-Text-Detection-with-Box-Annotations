import matplotlib.pyplot as plt

#折线图
x = ["0.5","1","5","10","15","20"]#点的横坐标
k1 = [200,300,500,800,1300,2000]#线1的纵坐标
k2 = [40,60,100,160,240,340]#线2的纵坐标
# plt.legend(loc=2,prop={'size':6})
plt.plot(x,k1,'s-',color = 'r',label="Annotation Cost")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="Collection Cost")#o-:圆形
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.xlabel("Data Scale(k)",size=15)#横坐标名字
plt.ylabel("Cost($)",size=15)#纵坐标名字
plt.legend(loc = "best",prop={'size':12})#图例
plt.savefig("filename.png")
# plt.show()