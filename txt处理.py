import numpy as np


with open(r"D:\vscode\network\requirements.txt",'r',encoding='utf-8')as file:
    with open(r"D:\vscode\network\requirements_1.txt",'w',encoding='utf-8')as file_1:
        lines = file.readlines()
        for line in lines:
            line=line.split('@')
            file_1.write(line[0])
        
