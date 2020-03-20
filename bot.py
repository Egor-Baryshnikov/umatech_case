
import telebot
from telebot import apihelper
from matplotlib.pyplot import imread
from bot_prediction import *

# Bot initializing
token = 'TOKEN'
apihelper.proxy = 'PROXY'

bot = telebot.TeleBot(token)

bot_working_dir = 'bot_imgs/'

# Help handler with task description
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "I can identify a person on Ufa-DinamoM soccer match video stream.\
                           Just upload the image (bounding box) as a photo.")

# Model usage with photo
@bot.message_handler(content_types=['photo', 'document'])
def photo(message):
    if message.content_type == 'photo':
        fileID = message.photo[-1].file_id
    else:
        fileID = message.document.file_id

    # download the file
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    # Notification if the image is small
    # height = message.photo[-1].height
    # width = message.photo[-1].width
    # print('h,w:', height, width)
    # if (height < 80) or (width < 40):
    #     bot.reply_to(message, 'The bounding box is too small :C. It is hard to guess, but I will try...')

    # save the file
    with open(bot_working_dir+"image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    # make prediction from the saved file
    img = torch.tensor(imread(bot_working_dir+"image.jpg") / 255, dtype=torch.float32)
    prediction_dict = classify_img(img.permute(2, 0, 1))

    # choose best class and send to user
    sorted_prediction = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)
    best_prediction = sorted_prediction[0]
    bot.reply_to(message, '{}'.format(best_prediction[0]))

bot.polling()