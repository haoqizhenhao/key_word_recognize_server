import requests
import platform
from utils.logging import LOG
from flask import Flask, request

# 加载关键词识别模块
import recognition

# 启用flask服务
if platform.system() == "Windows":
    slash = '\\'
else:
    platform.system() == "Linux"
    slash = '/'
app = Flask(__name__)
log = LOG()

kws = recognition.KWS(model_dir='model/CNN_L.pb')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        get_detail = request.form
        get_file = request.files
        print(get_file)
        try:
            way = get_detail['way']
        except Exception as e:
            log.error("[get_error]" + str(e))
            print('no right way')
            log.error(str(get_detail) + 'stat:no right way')
            return 'stat:no right way'

        try:
            wav_name = get_detail['wav']
        except Exception as e:
            log.error("[get_error]" + str(e))
            print('no right wav')
            log.error(str(get_detail) + 'stat:no right wav')
            return 'stat:no right wav'

        try:
            file_realtime = get_file['file']
            print('******', file_realtime)
        except Exception as e:
            log.error("[get_error]" + str(e))
            print('no right wav')
            log.error(str(get_detail) + 'stat:no right wav')
            return 'stat:no right wav'

        if way == 'file':
            try:
                return_table = kws.recognize_file(wav_name)
            except:
                log.error(str(wav_name) + 'stat:model deal error')
                print('model deal error !')
                return 'stat:model deal error'
            return return_table

        if way == 'real_time':
            try:
                return_table = kws.recognize_realtime(wav_stream=file_realtime)
            except:
                # log.error(str(wav_name) + 'stat:model deal error')
                print('model deal error !')
                return 'stat:model deal error'
            return return_table


if __name__ == "__main__":
    app.run(host='127.0.0.3', port=5000)
