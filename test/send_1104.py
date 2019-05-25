import pandas as pd
import requests
import time 
if __name__ == '__main__':

    wav_path = r'../wav/test1.wav'
    req = requests.post(url="http://127.0.0.1:5000/",
                        data={'wav': wav_path, 'way': 'KWS'})
    wenben = req.text
    print(wenben)
    # wb=pd.read_csv('0102_id_url.csv')
    # url_path=list(wb.image_path)[:50]
    # student_id=list(wb.student_id)
    # for i in range(len(url_path)):
    #     req = requests.post(url="http://127.0.0.1:5000/",data={'image_url':url_path[i],'student_id':str(student_id[i]),'way':'Compare'})
    #     wenben=req.text
    #     print(str(student_id[i]),wenben)


# import pickle as pk
# pk.dump({},open('../data/data.pkl','wb'))