from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Welcome to LangChain All purpose Q&A Bot!'

os.system("bot.py")

if __name__ == '__main__':
    app.run()
