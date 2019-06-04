import requests
import os
if __name__ == '__main__':

    # wav_path = r'../DATA\speech_commands_haifeng_v0.01\yes'
    # for wav in os.listdir(wav_path):
    #     req = requests.post(url="http://127.0.0.1:5000/", data={'wav': os.path.join(wav_path, wav), 'way': 'file'})
    #     wenben = req.text
    #     print(wenben)

    wav_path = r'wav/test1.wav'
    # with open(wav_path, 'rb') as wav_file:
    #     wav_data = wav_file.read()
    #
    # print(wav_data)
    # print(type(wav_data))
    req = requests.post(url="http://127.0.0.2:5000/", data={'wav': wav_path, 'way': 'file'})
    wenben = req.text
    print(wenben)