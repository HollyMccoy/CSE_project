import requests
import datetime

def post_db(sound,device, d, t):
    #sound class variable pass into function
    sound_class=sound
    #deviceID will also pass into function
    deviceID=device
    date_today = d
    time_now = t

    URL = "https://wb44syzvc9.execute-api.us-east-2.amazonaws.com/beta/sound"
    headers = {"Content-Type":"aplication/json"}
    params = {"payload":"payload???"}

    data = dict(deviceID=deviceID,date=str(datetime.date.today().strftime("%m/%d/%y")),UTC=str(datetime.datetime.now().strftime("%H:%M:%S")),soundID=sound_class)
    
    #data = dict(deviceID=deviceID,date=date_today,UTC=time_now,soundID=sound_class)

    response = requests.post(URL,json=data,headers= headers)

    print(response.text)

