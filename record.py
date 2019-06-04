import pyaudio,wave
from struct import pack
import requests

RATE = 16000
CHUNK_SIZE = 2000


def record_to_file(path, data, sample_width):
    "Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16,
                 channels=1,
                 rate=RATE,
                 input=True,
                 start=False,
                 # input_device_index=2,
                 frames_per_buffer=4000)
save_buffer = ''

print("* recording: ")
stream.start_stream()

while True:
    string_audio_data = stream.read(CHUNK_SIZE)
    print(string_audio_data)
    s_int = int.from_bytes(string_audio_data, byteorder='big')
    print(s_int)
    save_buffer += string_audio_data

    if len(save_buffer) >=16000:
        req = requests.post(url="http://127.0.0.3:5000/", data={'wav': save_buffer, 'way': 'real_time'}, files=save_buffer)
        wenben = req.text
        print(wenben)
        stream.stop_stream()
        break

# pa = pyaudio.PyAudio()
# stream = pa.open(format=pyaudio.paInt16,
#                  channels=1,
#                  rate=16000,
#                  input=True,
#                 frames_per_buffer=2000)
#
# save_buffer = ''
#
# wf = wave.open('haha.wav', 'wb')
# wf.setnchannels(1)
# wf.setsampwidth(2)
# wf.setframerate(16000)
# try:
#     while True:
#         string_audio_data = stream.read(1000)
#         save_buffer += string_audio_data
#         if len(save_buffer) >= 16000:#采样8次就保存
#             wf.writeframes(save_buffer)
#             break
# except:
#     wf.close()