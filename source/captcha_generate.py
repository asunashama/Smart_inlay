import random
import string

from captcha.image import ImageCaptcha

# characters为验证码上的字符集，10个数字加26个大写英文字母
# 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ str类型
characters = string.digits + string.ascii_uppercase

width, height, n_len, n_class = 170, 80, 4, len(characters)

# 生成一万张验证码
for i in range(100):
    # 设置验证码图片的宽度width和高度height
    # 除此之外还可以设置字体fonts和字体大小font_sizes
    generator = ImageCaptcha(width=width, height=height)
    # 生成随机的4个字符的字符串
    random_str = ''.join([random.choice(characters) for j in range(4)])
    img = generator.generate_image(random_str)

    # 将图片保存在目录文件夹下
    file_name = './train_img/ ' + random_str + '_ ' + str(i) + '.jpg'
    # 生成验证码
    img.save(file_name)
