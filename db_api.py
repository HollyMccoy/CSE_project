import requests
import datetime

def post_db(sound,device):
    #sound class variable pass into function
    sound_class=sound
    #deviceID will also pass into function
    deviceID=device
    
    #URL = "put api gateway URL here"
    headers = {"Content-Type":"aplication/json"}
    params = {"payload":"payload???"}

    data = dict(deviceID=deviceID,date=str(datetime.date.today().strftime("%m/%d/%y")),UTC=str(datetime.datetime.now().strftime("%H:%M:%S")),soundID=sound_class)

    response = requests.post(URL,json=data,headers= headers)

    print(response.text)


post_db("cat","holly")