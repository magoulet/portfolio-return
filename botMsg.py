import json
import os
import requests


def telegram_bot_sendtext(bot_message):

    curPath = os.path.dirname(os.path.abspath(__file__))
    configurationFile = curPath + '/MsgConfig.json'
    configuration = json.loads(open(configurationFile).read())

    bot_token = configuration["telegram"][0]["bot_token"]
    bot_chatID = configuration["telegram"][0]["bot_chatID"]
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()
