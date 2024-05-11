import streamlit as st
import asyncio
from collections import deque
import pandas as pd
import random
from twitchio.ext import commands
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('TOKEN')

# Function to generate a random color
def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Dictionary to store colors for each user
user_colors = {}

# Custom CSS for larger chat box, better visibility, and wider message column
st.markdown(
    """
    <style>
    .chat-box {
        height: 400px;
        overflow-y: scroll;
        font-size: 15px;
        padding: 10px;
        border-radius: 10px;  
        background-color: #282C34;
        color: #FFFFFF;
        margin-bottom: 20px;
    }
    .dataframe th, .dataframe td {
        text-align: left;
        font-size: 18px;
        padding: 10px;
    }
    .dataframe th {
        background-color: #21252B;
    }
    .dataframe {
        width: 100%;
    }
    .col-message {
        width: 70%;  /* Adjust this value as needed */
    }
    </style>
    """, unsafe_allow_html=True
)

# Title of the Streamlit application
st.title("Twitch Monitor")

# Input for the channel name
channel = st.text_input('Enter the channel name')

# Placeholder for the chat box and the data table
chat_box = st.empty()
chat_box.markdown('<div class="chat-box" id="chat-box"></div>', unsafe_allow_html=True)

df = pd.DataFrame(columns=['Pseudo', 'Message', 'Toxique', 'Très toxique', 'Obscène', 'Menace', 'Insulte', 'Haine identitaire'])
table = st.empty()

# Queue to store the last 10 messages
message_queue = deque(maxlen=20)

class Bot(commands.Bot):
    def __init__(self, channel_name):
        super().__init__(token=TOKEN, prefix='!', initial_channels=[channel_name])

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')

    async def event_message(self, message):
        global df
        print(f'{message.author.name}: {message.content}')

        # Assign a random color to the user if not already assigned
        if message.author.name not in user_colors:
            user_colors[message.author.name] = get_random_color()

        # Get the user's color
        user_color = user_colors[message.author.name]

        # Append message to queue and update the chat box
        colored_message = f'<span style="color:{user_color};">{message.author.name}:</span> {message.content}'
        message_queue.append(colored_message)
        messages_html = '<br>'.join(message_queue)
        chat_box.markdown(f'<div class="chat-box">{messages_html}</div>', unsafe_allow_html=True)
        
        # Append message to the dataframe with a random toxicity score
        new_row = pd.DataFrame([[message.author.name, message.content, random.random(), random.random(), random.random(), random.random(), random.random(), random.random()]], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Format the message column specifically and set decimal places for numeric columns
        df_styled = df.style.set_properties(subset=['Message'], **{'width': '300px', 'text-align': 'left'})
        df_styled = df_styled.format(subset=['Toxique', 'Très toxique', 'Obscène', 'Menace', 'Insulte', 'Haine identitaire'], formatter="{:.2f}")
        table.dataframe(df_styled)
                
        await self.handle_commands(message)

# Variable to control the bot's state
bot_running = False

# Function to run the bot
async def run_bot(channel_name):
    global bot_running
    bot = Bot(channel_name)
    bot_running = True
    try:
        await bot.start()
    finally:
        bot_running = False

# Function to start the bot in a new thread
def start_bot(channel_name):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(run_bot(channel_name))
    loop.run_forever()

# Stop the bot
def stop_bot():
    global bot_running
    bot_running = False
    for task in asyncio.all_tasks():
        task.cancel()
        
# Execute the function to start or stop the bot based on the button pressed
if st.button('Start Bot') and not bot_running:
    if channel:
        start_bot(channel)
    else:
        st.error("Please enter a channel name.")
elif st.button('Stop Bot') and bot_running:
    stop_bot()
