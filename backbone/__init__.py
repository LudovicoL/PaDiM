from .mvtec import *
from .AITEX import *
from .CustomDataset import *

import requests
import json

def myPrint(string, filename):
    print(string)
    filename.write(string + '\n')


def telegram_bot_sendtext(bot_message):
    """
    Send a notice to a Telegram chat. To use, create a file "tg.ll" in the main folder with this form:
    {
    "token": "",    <-- bot token 
    "idchat": ""    <-- your chat id
    }
    """
    try:
        with open('./tg.ll') as f:
            data = json.load(f)
    except:
        print("ERROR: Can't send message on Telegram. Configure the \'./tg.ll\' file or set args.telegram=False.")
        return
    bot_token = data['token']
    bot_chatID = data['idchat']
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return str(response)


def precision(tp, fp):
    return tp/(tp + fp)

def sensitivity(tp, fn):    # True Positive Rate
    return tp/(tp + fn)

def FPR(fp, tn):            # False Positive Rate
    return fp/(fp + tn)

def F_score(precision, sensitivity, beta):
    return (1 + beta**2) * ((precision * sensitivity)/(beta**2 * precision + sensitivity))
