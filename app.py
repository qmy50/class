import streamlit as st

# 设置页面标题
st.title("我的第一个 Streamlit 应用")
st.markdown('I LOVE **YOU** TOO')

# 显示文本
st.write("hello")
st.write('World !')

print('start run')
pressed1=st.button('button 1')
print('pressed1',pressed1)
if pressed1:
    st.write('YES ')

pressed2=st.button('button 2')
print('pressed2',pressed2)