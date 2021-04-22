import requests
import datetime

def post_db(sound,device, t):
    #sound class variable pass into function
    sound_class=sound
    #deviceID will also pass into function
    deviceID=device
    #time passed into function
    time_now = t

    URL = # insert API Gateway URL for triggering Lambda here
    headers = {"Content-Type":"aplication/json"}
    params = {"payload":"payload???"}

    data = dict(deviceID=deviceID,date=str(datetime.date.today().strftime("%m/%d/%y")),UTC=time_now,soundID=sound_class)
    
    response = requests.post(URL,json=data,headers= headers)

    print(response.text)

