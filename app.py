from flask import Flask, render_template, request, url_for, flash, redirect
from threading import Thread
from text import *
# from voice import *
# from hand import *
# from hand_2 import *
import multiprocessing


app = Flask(__name__)
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'
v = None
h = None


def voice_thread():
    text("play blank space by taylor swift")
    # voice()


def hand_thread():
    text("play blank space by taylor swift")
    # hand_2()


def end_threads():
    global v
    global h
    if v is not None:
        v.terminate()
        v = None
    if h is not None:
        h.terminate()
        h = None


@app.route('/')
def index():
    end_threads()
    return render_template('index.html')


@app.route('/trial')
def trial():
    end_threads()
    return render_template('trial.html')


@app.route('/trial/text-command', methods=('GET', 'POST'))
def textCommand():
    end_threads()
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
    end_threads()
    return render_template('thanks.html')


@app.route('/trial/hand-gesture')
def handGesture():
    global h
    if h is None:
        h = multiprocessing.Process(target=hand_thread, args=())
        # h= Thread(target=hand_thread)
        h.start()
    return render_template('hand.html')


@app.route('/trial/voice-command')
def voiceCommand():
    global v
    if v is None:
        v = multiprocessing.Process(target=voice_thread, args=())
        # v = Thread(target=voice_thread)
        v.start()
    return render_template('voice.html')
