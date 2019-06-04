import requests
import os
import json
# from base64 import b64encode, b64decode

if __name__ == '__main__':
    wav_path = r'wav/test1.wav'
    # with open(wav_path, 'rb') as wav_file:
    #     # wav_data = wav_file.read()
    #     requests.post(url="http://127.0.0.3:5000/", data=wav_file)
    files = {'file': open(wav_path, 'rb')}
    print(files)
    req = requests.post(url="http://127.0.0.3:5000/", data={'wav': wav_path, 'way': 'real_time'}, files=files)
    wenben = req.text
    print(wenben)
