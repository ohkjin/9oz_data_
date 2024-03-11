from rembg import remove

input_path = 'C:/Users/user/test/img'
output_path = 'C:/Users/user/test/img/nukki'

def getNukki(filename):
    with open(input_path+'/'+filename, 'rb') as i:
        with open(output_path+'/'+filename, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)