import streamlit as st
from PIL import Image

# from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error

# 载入数据
data = np.loadtxt(r'D:\vscode\machine_learning\机器学习\机器学习\回归\线性回归以及非线性回归\house_price.txt',dtype=int,delimiter=',')     
print(data)
x_data=data[:,0]
y_data=data[:,1]
plt.scatter(x_data,y_data,c='b',marker='o')
plt.xlabel('Area')
plt.ylabel('Price')


lr=0.0001
b=0
k=0
epoch=50

def loss(k,b,x_data,y_data):
    res=0
    for i,j in zip(x_data,y_data):
        res=res+(j-(k*i+b))**2
    res=res/(2*len(x_data))
    return res
fig, axes = plt.subplots(2, 5, figsize=(15, 5))

def gd(b,k):
    num=0
    for i in range(epoch):
        for j in range(len(x_data)):
            b_cur=b-lr*(k*x_data[j]+b-y_data[j])/len(x_data)
            k_cur=k-lr*(k*x_data[j]+b-y_data[j])/len(x_data)*x_data[j]
            b=b_cur
            k=k_cur
        if i%5==0:
            axes[int(num>=5),num%5].scatter(x_data,y_data,c='b',marker='o')
            axes[int(num>=5),num%5].plot(x_data,k*x_data+b) 
            if num==5:
                axes[int(num>=5),num%5].set_xlabel('Area')
                axes[int(num>=5),num%5].set_ylabel('Price')
            num=num+1
    fig.suptitle('Gradient Descent Process')
    plt.show()
    return b,k
my_b,my_k=gd(b,k)
# print(f"斜率为：{my_k} 截距为:{my_b}")
# print('损失函数：'+str(loss(my_k,my_b,x_data,y_data)))
# ans=my_k*130+my_b
# print('结果为：'+str(ans))

st.write('机器学习作业展示：')

if "step" not in st.session_state:
    st.session_state['step']=1

def goto_step(step_num:int):
    st.session_state['step']=step_num

if st.session_state.step==1:
    st.title('算法选择')
    st.button(label='梯度下降——房价问题',on_click=goto_step,args=((2,)))
    st.button(label='SVM——鸢尾花分类',on_click=goto_step,args=((3,)))


if st.session_state.step==2:
    st.title('梯度下降')
    st.button(label='Go prev',on_click=goto_step,args=((1,)))
    example="""iris = datasets.load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target) """


    st.code(example,language='python')

    st.divider()

    slider_val=st.slider(label='请输入房屋面积：',
          min_value=50,
          max_value=150,
          value=0,
          step=1)
    ans=my_k*slider_val+my_b

    st.write("预测房价结果为：",'%.2f' % ans)


if st.session_state.step==3:
    st.title('支持向量机')
    st.button(label='Go prev',on_click=goto_step,args=((1,)))


# example="""iris = datasets.load_iris()
# x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target) """


# st.code(example,language='python')

# st.divider()

# slider_val=st.slider(label='请输入房屋面积：',
#           min_value=50,
#           max_value=150,
#           value=0,
#           step=1)
# ans=h.my_k*slider_val+h.my_b

# st.write("预测房价结果为：",'%.2f' % ans)

image_1=Image.open(r'D:\vscode\machine_learning\06966bbd48e8c52c4c91b7b11c724da1.png')

st.image(image_1)

