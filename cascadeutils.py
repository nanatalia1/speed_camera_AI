import os

def generate_negative_description_file():
    path = r"C:\Users\48516\Desktop\AIproject\negative"
    with open('neg.txt', 'w') as f:
        for filename in os.listdir(path):
            f.write('negative/'+filename+'\n')