from PIL import Image

model_filename = ''
result_filename = ''
background_filename = ''

img = Image.open(model_filename)
img = img.convert('RGBA')
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)

background = Image.open(background_filename)
background.paste(img, (200, 200), img)
background.save(result_filename, 'PNG')

# Subprocess ImageMagic Tool
# 'mogrify +noise Gaussian -blur 20 sample.png'
