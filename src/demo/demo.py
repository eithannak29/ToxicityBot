import streamlit as st
import asyncio
from collections import deque
import pandas as pd
from twitchio.ext import commands
import os
from dotenv import load_dotenv
from utils import query , get_random_color

load_dotenv()

TOKEN_TWITCH = os.getenv('TOKEN_TWITCH')
TOKEN_TWITCH_USER = os.getenv('TOKEN_TWITCH_USER')

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
    .message {
        padding: 5px;
        border-bottom: 1px solid #444;
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
        width: 66.67%; /* Adjusted width to span two-thirds */
    }
    </style>
    """, unsafe_allow_html=True
)

# Title of the Streamlit application
st.title("Twitch Monitor 🤖")

# Input for the channel name
channel = st.text_input('Enter the channel name')

# Option to select AI model
ai_model = st.selectbox('Select AI Model', ['Transformer', 'Logistic Regression'])

# Placeholder for the chat box and the data table
chat_box = st.empty()
chat_box.markdown('<div class="chat-box" id="chat-box"></div>', unsafe_allow_html=True)

df = pd.DataFrame(columns=['Username', 'Message', 'Toxic', 'Severe toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate'])
table = st.empty()

# Queue to store the last 20 messages
message_queue = deque(maxlen=50)

class Bot(commands.Bot):
    def __init__(self, channel_name):
        super().__init__(token=TOKEN_TWITCH_USER, prefix='!', initial_channels=[channel_name])

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
        colored_message = f'<div class="message"><span style="color:{user_color};">{message.author.name}:</span> {message.content}</div>'
        message_queue.append(colored_message)
        messages_html = '<br>'.join(message_queue)
        chat_box.markdown(f'<div class="chat-box">{messages_html}</div>', unsafe_allow_html=True)
        
        # Analyze the message using the selected AI model
        if ai_model == 'Transformer':
            response = query({"inputs": message.content})
        
        new_row = pd.DataFrame([[message.author.name, message.content, response['toxic'], response['severe_toxic'], response['obscene'], response['threat'], response['insult'], response['identity_hate']]], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Apply conditional text color formatting
        df_styled = df.style.applymap(get_color, subset=['Toxic', 'Severe toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']).format(subset=['Toxic', 'Severe toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate'], formatter="{:.2f}")
        table.dataframe(df_styled, width=1000)
                
        await self.handle_commands(message)

def get_color(val):
    red = int(val * 255)
    green = int((1 - val) * 255)
    return f'color: rgb({red},{green},0)'

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

# Organize the layout with columns
col1, col2 = st.columns(2)

with col1:
    if st.button('Start Bot') and not bot_running:
        if channel:
            start_bot(channel)
        else:
            st.error("Please enter a channel name.")

with col2:
    if st.button('Stop Bot') and bot_running:
        stop_bot()

# Message to inform the user about the bot's status
if bot_running:
    st.success("Bot is running")
else:
    st.warning("Bot is stopped")
