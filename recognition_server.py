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

        if way == 'KWS':
            try:
                return_table = kws.rec(wav_name)
            except:
                log.error(str(wav_name) + 'stat:model deal error')
                print('model deal error !')
                return 'stat:model deal error'

            return return_table


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
