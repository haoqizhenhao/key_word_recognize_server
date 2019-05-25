import requests
import os
if __name__ == '__main__':

    wav_path = r'../DATA\speech_commands_haifeng_v0.01\yes'
    for wav in os.listdir(wav_path):
        req = requests.post(url="http://127.0.0.1:5000/", data={'wav': os.path.join(wav_path, wav), 'way': 'KWS'})
        wenben = req.text
        print(wenben)