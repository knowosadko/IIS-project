from time import sleep
import random
from furhat_remote_api import FurhatRemoteAPI
from numpy.random import randint
import threading
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

class Thread_B(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.name = "FaceDetectionThread"
        
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

    def emotion_categorizer(emote):
        if emote == "angry" or "disgust" or "fear" or "sad":
            return "negative"
        elif emote == "happy":
            return "positive"
        else:
            return "neutral"

    def main_tree():
        set_persona('Amany')
        furhat.set_voice(name='Matthew')
        
        current_emotion = "currentEmotion"
        current_emotion_categorized = emotion_categorizer("current_emotion")
        match current_emotion_categorized:
            case "positive":
                first_pos()
            case "negative":
                first_neg()
            case "neutral":
                first_pos()

    def first_pos():
        bsay("Hello! how are you today? You seem kind of happy, am i wrong?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                mistake_pos()
            elif "no" in message_lower:
                second_pos()
                
    def first_neg():
        bsay("Hello! how are you today? You seem kind of sad, am i wrong?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                mistake_neg()
            elif "no" in message_lower:
                second_neg()
                
    def mistake_pos():
        bsay("my mistake, are you sad?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                mistake_pos()
            elif "no" in message_lower:
                first_neutral()
                
    def mistake_neg():
        bsay("my mistake, are you happy?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                mistake_pos()
            elif "no" in message_lower:
                first_neutral()
                
    def second_pos():
        bsay("Thats Great! Feeling happy is awesome. Can i offer you a drink?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                drink_emotion()
            elif "no" in message_lower:
                no_drink()
        
    def second_neg():
        bsay("We all have our ups and downs. Can i offer you a drink to perhaps make you feel better?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                drink_emotion()
            elif "no" in message_lower:
                no_drink()
        
    def first_neutral():
        bsay("I understand. you are neither sad or happy. Thats ok. Can i offer you a drink to perhaps make you happy?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                drink_emotion()
            elif "no" in message_lower:
                no_drink()
        
    def no_drink():
        bsay("What are you doing in a bar then? Either leave or have a drink. Are you sure you dont want to have a drink?")
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                no_drink()
            elif "no" in message_lower:
                drink_emotion()
        
        
    def drink_emotion():
        bsay("Okay, let me take a look at you and try to figure our how youre feeling currently....")
        sleep(2.5)
        drink_offer()
        
    def drink_offer():
        #TODO: ADD A FUNCTION TO GET EMOTION IDK HOW WE ARE SUPPOSED TO GET IT IN THE FIRST PLACE SO I WAIT WITH IT.
        emotion_seven = "placeholder"
        cocktails = ["Mojito", "Moscow Mule", "Aperol", "Martini", "Daiquiri", "Margarita", "Negroni"]
        cocktail = random.choice(cocktails)
        bsay("Ill offer you our special ", emotion_seven, " ", cocktail, " That you will not find anywhere else. Would you like that?" )
        while True:
            # Speak and listen
            result = furhat.listen()
            message_lower = result.message.lower()
            if "yes" in message_lower:
                drink_accepted()
            elif "no" in message_lower:
                drink_emotion()

    def drink_accepted():
        bsay("Great")
    
    
    

if __name__ == '__main__':
    main_tree()
    idle_animation()
