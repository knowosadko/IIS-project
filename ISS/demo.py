from time import sleep
from furhat_remote_api import FurhatRemoteAPI
from numpy.random import randint

FURHAT_IP = "127.0.1.1"

furhat = FurhatRemoteAPI(FURHAT_IP)
furhat.set_led(red=100, green=50, blue=50)


FACES = {
    'Loo'    : 'Patricia',
    'Amany'  : 'Nazar'
}

VOICES_EN = {
    'Loo'    : 'BellaNeural',
    'Amany'  : 'CoraNeural'
}

VOICES_NATIVE = {
    'Loo'    : 'SofieNeural',
    'Amany'  : 'AmanyNeural'
}

def idle_animation():
    furhat.gesture(name="GazeAway")
    gesture = {"frames" : 
        [{
            "time" : [0.33],
            "persist" : True,
            "params": {
                "NECK_PAN"  : randint(-4,4),
                "NECK_TILT" : randint(-4,4),
                "NECK_ROLL" : randint(-4,4),
            }
        }],

    "class": "furhatos.gestures.Gesture"
    }
    furhat.gesture(body=gesture, blocking=True)

def LOOK_BACK(speed):
    return {
    "frames": [
        {
            "time": [
                0.33 / speed
            ],
            "persist": True,
            "params": {
                'LOOK_DOWN' : 0,
                'LOOK_UP' : 0,
                'NECK_TILT' : 0
            }
        }, {
            "time": [
                1 / speed
            ],
            "params": {
                "NECK_PAN": 0,
                'LOOK_DOWN' : 0,
                'LOOK_UP' : 0,
                'NECK_TILT' : 0
            }
        }
    ],
    "class": "furhatos.gestures.Gesture"
    }

# DO NOT CHANGE
def LOOK_DOWN(speed=1):
    return {
    "frames": [
        {
            "time": [
                0.33 / speed
            ],
            "persist": True,
            "params": {
#                'LOOK_DOWN' : 1.0
            }
        }, {
            "time": [
                1 / speed
            ],
            "persist": True,
            "params": {
                "NECK_TILT": 20
            }
        }
    ],
    "class": "furhatos.gestures.Gesture"
    }

def set_persona(persona):
    furhat.gesture(name="CloseEyes")
    furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
    sleep(0.3)
    furhat.set_face(character=FACES[persona], mask="Adult")
    furhat.set_voice(name=VOICES_EN[persona])
    sleep(2)
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)

# Say with blocking (blocking say, bsay for short)
def bsay(line):
    furhat.say(text=line, blocking=True)

def demo_personas():
    set_persona('Amany')
    furhat.set_voice(name='Matthew')
    furhat.say(text = "Hello Hasan, how are you today")
    while True:
        # Speak and listen
        result = furhat.listen()

        message_lower = result.message.lower()

        if "hello" in message_lower:
            furhat.say(text ="Im fine, thank you")
        elif "sexy" in message_lower:
             furhat.say(text ="Thank you Hasan, thats very kind of you")
             furhat.gesture(body={
                "frames": [
                    {
                        "time": [0.33],
                        "params": {
                            "BLINK_LEFT": 1.0
                        }
                    },
                    {
                        "time": [0.67],
                        "params": {
                            "reset": True
                        }
                    }
                ],
                "class": "furhatos.gestures.Gesture"
            })

        elif "weather" in message_lower:
            furhat.say(text ="Do I look like a weather app?")
        elif "rap music" in message_lower:
            furhat.say(text ="I always listen to rap music, do you want me to play a song for you? ")
        elif "yes" in message_lower:
            furhat.say(url="https://dl.dropbox.com/scl/fi/53ysdc1f482je69nxsoz0/Future_ft_Juice_Wrld_-_Fine_China.wav?rlkey=ao06qeqavh3si5rdvtgagnubo&", lipsync=True)






   
    

if __name__ == '__main__':
    demo_personas()
    idle_animation()
