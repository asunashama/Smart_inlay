from captcha.image import ImageCaptcha
import numpy as np 
import random
import string 

#characters为验证码上的字符集，10个数字加26个大写英文字母
#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ str类型
characters=string.digits+string.ascii_uppercase

width,height,n_len,n_class=170,80,4,len(characters)

#设置验证码图片的宽度widht和高度height
#除此之外还可以设置字体fonts和字体大小font_sizes
generator=ImageCaptcha(width=width,height=height)

#生成随机的4个字符的字符串
random_str=''.join([random.choice(characters) for j in range(4)])

#生成验证码
img=generator.generate_image(random_str)

file_name=random_str+'.jpg'
img.save(file_name)
