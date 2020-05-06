# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:47:47 2020

@author: Wang
批量处理图片
转灰度，修改尺寸等
"""

from PIL import Image
import os
 
file_dir = 'E:/code/matlab/车牌/getword/G/'

out_dir = 'E:/code/python/chepai/dataset/carplate/ann/G/'

a = os.listdir(file_dir)

size_m = 20
size_n = 20
for i in a:
    print(i)
    I = Image.open(file_dir+i)
    
    L = I.resize((size_m, size_n),Image.ANTIALIAS)#修改尺寸
    
    #L = I.convert('L')#灰度
    
    
    L.save(out_dir+i)
