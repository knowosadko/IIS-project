from time import sleep
import random
from furhat_remote_api import FurhatRemoteAPI
from numpy.random import randint
import globals
from texts import *
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

#DEBUG PURPOUSES
emotion = "happy"

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

model = SentenceTransformer('distilbert-base-nli-mean-tokens')


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
    furhat.set_face(character=FACES[persona], mask="Adult")
    furhat.set_voice(name=VOICES_EN[persona])
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)

# Say with blocking (blocking say, bsay for short)
def bsay(line):
    furhat.say(text=line, blocking=True)

def get_emotion(mode="complete"):
    globals.semaphor.acquire()
    emote = globals.emotion[0]
    globals.semaphor.release()
    if mode == "valence":
        if emote in ["angry", "disgust", "fear", "sad"]:
            return "negative"
        elif emote == "happy":
            return "positive"
        elif emote== "surprise":
            return "neutral"
        else:
            return "neutral"
    elif mode=="reduced":
        if emote in ["disgust","fear","surprise"]:
            return "neutral"
        else:
            return emote
    else:
        return emote
        
            
 
 
def main_tree():
    set_persona('Amany')
    furhat.set_voice(name='Matthew')
    emotion = None
    while emotion == None: 
        globals.semaphor.acquire()
        emotion = globals.emotion
        globals.semaphor.release()
        sleep(1)  
    emotion = get_emotion(mode="reduced")
    if globals.debug:
        print("Current emotion:"+emotion)
    if emotion=="neutral":
        first_neutral(emotion)
    else:
        first_emotion(emotion)        

def get_text(lines):
    line = random.choice(lines)
    return line+" "

def listen():
    result = furhat.listen()
    while result.message == 'NOMATCH':
        result = furhat.listen()
    return result.message.lower()
    
def first_neutral(emotion):
    # TODO: Persona should smile here.
    bsay(get_text(GREATING)+" I see you are neither sad or happy. Thats ok. Can i offer you a drink to perhaps make you happy?")
    message = listen()
    if similar(message,"Yes, please"):
        drink_emotion(emotion)
    elif similar(message,"No I do not want a drink."):
        no_drink()

def first_emotion(emotion):
    # TODO: Persona should smile here.
    text = get_text(GREATING) + get_text(YOU_SEEM_KIND_OF) + emotion+" " + get_text(AINT_YOU)
    bsay(text)
    message = listen()
    if similar(message,"Yes, I do.") or similar(message,f"Yes, I am {emotion}."):
         second(emotion)
    elif similar(message, "No, I don't.") or similar(message,f"No I am not {emotion}"):
         emotion = mistake()
         second(emotion)
    else:
        bsay("R2D2 sounds: BEEP BOOP BIBIBIBIPOB")

def mistake():
    bsay("My bad, how do you feel then?")
    emotion = None
    while emotion == None:
        response = listen()
        if similar(response, "I am happy."):
            emotion = "happy"
        elif similar(response,"I am sad."):
            emotion = "sad"
        elif similar(response,"I am angry."):
            emotion = "angry"
        elif similar(response,"I am neutral"):
            emotion = "neutral"
        else:
            bsay("Could you repeat, I don't understand.")
    return emotion
     

def second(emotion):
    text = None
    if emotion == "happy":# TODO: Add other emotions, angry, happy, neutral, sad 
        text = get_text(GREAT) + get_text(HAPPY_IS_GOOD) + get_text(DRINK_OFFER)
    elif emotion=="sad":
        text = get_text(SYMPATHIZE_SAD) + get_text(DRINK_OFFER_SAD)
    elif emotion=="angry":
        text = get_text(SYMPATHIZE_ANGRY) + get_text(DRINK_OFFER_SAD)
    elif emotion == "neutral":
        first_neutral(emotion)
        return
    bsay(text)
    message = listen()
    if similar(message,"Yes."):
        drink_emotion(emotion)
    elif similar(message, "I don't want a drink."):
        no_drink()
             
def no_drink():
    text = get_text(WHY_IN_BAR) + get_text(DRINK_OFFER_2)
    bsay(text)
    # Speak and listen
    message = listen()
    if similar(message,"No, I do not want a drink."):
        no_drink()
    elif similar(message,"Yes, I want a drink."):
        drink_emotion()
    
def drink_emotion(emotion):
    bsay("Okay, let me take a look at you and try to figure our how youre feeling currently....")
    emotion = get_emotion(mode="reduced")
    # Add animation look down to up
    sleep(3)
    drink_offer(emotion)
    
def drink_offer(emotion):# TODO we can change it to be more dramatic
    cocktails = ["Mojito", "Moscow Mule", "Aperol", "Martini", "Daiquiri", "Margarita", "Negroni"]
    cocktail = random.choice(cocktails)
    bsay(f"Ill offer you our special {emotion} {cocktail} That you will not find anywhere else. Would you like that?" )
    done = False
    while not done:
        # Speak and listen
        message = listen()
        if similar(message,"Yes, please."):
            drink_accepted(emotion,cocktail)
            sleep(5)
            text = "Another round?"
            bsay(text)
            message = listen()
            if similar(message,"No, thank you."):
                bsay("I hope you had a good time. See you!")
                # TODO: persona should smile
                done = True
            # Assume any other response is yes.
        elif similar(message,"No, I do not want it."):
            drink_emotion(emotion)
        
        
        

def drink_accepted(emotion,cocktail):
    bsay(get_text(HAND_DRINK))
    sleep(7)
    # emotion detect 
    emotion = get_emotion(mode="complete")
    if emotion=="disgust":
        # TODO: express sadness
        bsay("I see you don't like it. Maybe it is not your style of the cocktail, I will try to make next one better.")
    elif emotion=="surprise":
        # TODO: express curiosity
        bsay("I see surprise on your face is it good or bad surprise.")
        message = listen()
        if similar(message,"It is good surprise."):
             bsay("Glad to hear.")
        elif similar(message, "It is bad surprise."):
            bsay("Sorry to hear, next cocktail will be better.")
    else:    
        bsay(get_text(LIKE_IT))
        message = listen()
        if similar(message, "Yes, I like it."):
            bsay("Glad to hear it, enjoy it.")
        elif similar(message, "No, I don't like it."):
            bsay("Sorry, next one will be better.")
        else:
            bsay("Enjoy it.")
    current_emotion = get_emotion(mode="valence")
    sleep(2)
    if current_emotion == "negative":
        bsay("I see, the drink did not cheer you up. Maybe you want to talk?")
    else:
        bsay("Now that you have a drink. Maybe we can talk?")
    message = listen()
    if similar(message,"yes"):
        free_conversation(current_emotion,cocktail)
        bsay("If you don't want to talk, it is fine.")
    else:
        bsay("Alright, no worries if you want to talk I am here.")
    
def free_conversation(emotion,cocktail):
    context = f"Context: You are barman that just served me a {cocktail} and I am a customer. I am feeling {emotion}" \
                "and would like to talk with you. Give only your lines, don't greet me."
    client = OpenAI(api_key="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",)
    chat_completion = client.chat.completions.create(messages=[
        {"role": "user","content": f"{context} Please start the conversation.",}], model="gpt-3.5-turbo",)
    text = chat_completion.choices[0].message.content
    bsay(text)
    response = listen()
    while not similar(response,"I don't want to talk."):
        chat_completion = client.chat.completions.create(messages=[
        {"role": "user","content": response,}], model="gpt-3.5-turbo",)
        bsay(chat_completion.choices[0].message.content)
        response = listen()
        
def similar(sentance, meaning):
    sentence_embeddings = model.encode([sentance, meaning])
    similarity = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1])
    if similarity.item() > 0.75:
        return True
    else:
        return False
    
if __name__ == '__main__':
    main_tree()
    idle_animation()
