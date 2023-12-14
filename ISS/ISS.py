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

def main_tree():
    set_persona('Amany')
    furhat.set_voice(name='Matthew')
    
    current_emotion = "currentEmotion"
    match current_emotion:
        case "Angry":
            first_angry()
        case "Disgust":
            first_disgust()
        case "Fear":
            first_fear()
        case "Happy":
            print("placeholder")
        case "Neutral":
            print("placeholder")
        case "Sad":
            print("placeholder")
        case "Surprised":
            print("placeholder")

def first_angry():
    furhat.say(text = "Hello! how are you today? You seem kind of frustrated")
    while True:
        # Speak and listen
        result = furhat.listen()
        message_lower = result.message.lower()
        
        if "good" in message_lower:
            furhat.say(text ="Oh okay, My mistake then. Can i offer you something? ")
        elif "bad" in message_lower:
             furhat.say(text ="Im sorry, can i offer you a drink to cheer you up?")
             #TODO: add yes or no etc blabla
             
def first_disgust():
    furhat.say(text = "Hello! how are you today? You seem kind of out of place")
    while True:
        # Speak and listen
        result = furhat.listen()
        message_lower = result.message.lower()
        
        if "good" in message_lower:
            furhat.say(text ="Oh okay, My mistake then. Can i offer you something?")
        elif "bad" in message_lower:
            furhat.say(text ="Im sorry, can i offer you a drink to cheer you up?")
            #TODO: add yes or no etc blabla
            
def first_fear():
    furhat.say(text = "Hello! how are you today? You seem kind of scared")
    while True:
        # Speak and listen
        result = furhat.listen()
        message_lower = result.message.lower()
        
        if "good" in message_lower:
            furhat.say(text ="Oh okay, My mistake then. Can i offer you something?")
        elif "bad" in message_lower:
            furhat.say(text ="Dont be scared, can i offer you a drink to make you feel more comfortable?")
            #TODO: add yes or no etc blabla
    
def first_happy():
    furhat.say(text = "Hello! how are you today? You seem happy")
    while True:
        # Speak and listen
        result = furhat.listen()
        message_lower = result.message.lower()
        
        if "good" in message_lower:
            furhat.say(text ="That makes me glad. Can i offer you a drink?")
        elif "bad" in message_lower:
            furhat.say(text ="Huh, you dont seem like youre down. Can i offer you a drink to make you feel better?")
            #TODO: add yes or no etc blabla

def first_neutral():
    furhat.say(text = "Hello! how are you today?")
    while True:
        # Speak and listen
        result = furhat.listen()
        message_lower = result.message.lower()
        
        if "good" in message_lower:
            furhat.say(text ="That makes me glad. Can i offer you a drink?")
        elif "bad" in message_lower:
            furhat.say(text ="Can i offer you a drink to make you feel better?")
            #TODO: add yes or no etc blabla

def first_sad():
    furhat.say(text = "Hello! how are you today? You seem kind of sad")
    while True:
        # Speak and listen
        result = furhat.listen()
        message_lower = result.message.lower()
        
        if "good" in message_lower:
            furhat.say(text ="Oh okay, My mistake then. Can i offer you a drink?")
        elif "bad" in message_lower:
            furhat.say(text ="Dont be sad. Let me offer you a drink to make you feel better.")
            #TODO: add yes or no etc blabla
            
def first_surprised():
    furhat.say(text = "Hello! how are you today? You seem surprised")
    while True:
        # Speak and listen
        result = furhat.listen()
        message_lower = result.message.lower()
        
        if "good" in message_lower:
            furhat.say(text ="Can i offer you a drink?") #TODO: what do you say if someone is surprised
        elif "bad" in message_lower:
            furhat.say(text ="Let me offer you a drink to make you feel better.")
            #TODO: add yes or no etc blabla


   
    

if __name__ == '__main__':
    main_tree()
    idle_animation()
