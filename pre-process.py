import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext('/home/guju/Desktop/test3.png',detail=0,paragraph=True)

for i in result:
    print(i)