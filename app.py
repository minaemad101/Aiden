from flask import Flask, render_template, request, url_for, flash, redirect
from threading import Thread
from text import *
import time
from voice import *
from hand import *
# import voice here


app = Flask(__name__)
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'


def voice_thread():
    # call voice function here
    # text("play blank space by taylor swift")
    # time.sleep(2.4)
    # text("play amira by wegz")
    # time.sleep(2.4)
    # text("play anything by amr diab")
    voice()

def hand_thread():
    hand()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/trial')
def trial():
    return render_template('trial.html')


@app.route('/trial/text-command', methods=('GET', 'POST'))
def textCommand():
    if request.method == 'POST':
        command = request.form['command']
        if not command:
            flash('Title is required!')
        else:
            text(command)
            return redirect(url_for('thanks'))
    return render_template('text.html')


@app.route('/trial/thanks')
def thanks():
    return render_template('thanks.html')


@app.route('/trial/hand-gesture')
def handGesture():
    t = Thread(target=hand_thread)
    t.start()
    return render_template('hand.html')


@app.route('/trial/voice-command')
def voiceCommand():
    t = Thread(target=voice_thread)
    t.start()
    return render_template('voice.html')
