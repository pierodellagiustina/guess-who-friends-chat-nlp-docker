from flask import Flask, render_template, request, url_for, redirect, session
import pandas as pd
import functionalities.functionalities as f
import config.config as c
import json


# Initialize app
app = Flask(__name__)
app.secret_key = 'gruppotelegram2'

# load the data
dataset = f.load_data()

# load the embedding vocab
embedding_vocab = f.load_embedding_vocab()

# load the model
nn_classifier = f.load_nn_model()


# home page
@app.route('/', methods=['GET', 'POST'])
def index():

    # Initialize the cumulative scores
    if 'cumul_score_user' not in session:
        session['cumul_score_user'] = 0
    if 'cumul_score_nn' not in session:
        session['cumul_score_nn'] = 0

    # Initialize attempts counter
    if 'attempts_counter' not in session:
        session['attempts_counter'] = 0

    # Read randomly chosen rows
    # todo: make num_samples dynamic
    if 'df' not in session:
        # randomly select the sample
        df = dataset.sample(c.NUM_SAMPLES)
        # turn it into json
        df = df.to_json()
        # add it to session
        session['df'] = df

    df = pd.read_json(session['df'])

    # Set up variables
    messages = df.message
    senders = df.sender
    sender_mapper = c.SENDER_MAPPER_REV
    cumul_score_nn = session['cumul_score_nn']
    cumul_score_user = session['cumul_score_user']
    attempts_counter = session['attempts_counter']

    # work out the NN predictions todo: move all this to the scoring section
    nn_distributions = []  # todo: implement this so that it shows the full distribution in the score page
    nn_chosen_senders = []
    score_nn = 0

    for i, m in enumerate(messages):
        true_sender = senders.values[i]
        nn_chosen_sender, nn_full_distribution = f.get_nn_predictions(sentence=m, embedding_vocab=embedding_vocab,
                             model=nn_classifier, sender_mapper_reverse=c.SENDER_MAPPER_REV)

        nn_chosen_senders.append(nn_chosen_sender)
        nn_distributions.append(nn_full_distribution)

        if nn_chosen_sender == true_sender:
            score_nn += 1

    df['nn_pred'] = nn_chosen_senders
    session['df'] = df.to_json()  # replace the df in the session

    if request.method == 'GET':  # request.method will tell me if it is a GET or POST request
        # Welcome page
        return render_template(
            'index.html', messages=messages, sender_mapper=sender_mapper,
            cumul_score_user=cumul_score_user, attempts_counter=attempts_counter, cumul_score_nn=cumul_score_nn)

    elif request.method == 'POST':
        # Get the request form
        request_form = request.form

        # If it is a request to refresh messages, do so
        if 'refresh' in request_form:
            session.pop('df')
            return redirect(url_for('index'))
        else:
            # Get user predictions from POST
            user_preds = f.get_user_preds(messages=messages, request_form=request_form)
            # Compute user score. todo: move this to the scoring section
            score_user = f.compute_user_score(user_preds, senders)
            # Update attempts counter
            session['attempts_counter'] += len(df)
            # convert to json before passing to redirect
            user_preds = json.dumps(user_preds)

            return redirect(url_for('score', score_user=score_user, user_preds=user_preds, score_nn=score_nn))


@app.route('/score/<score_user>/<user_preds>/<score_nn>')
def score(score_user, user_preds, score_nn):

    # Split input string and cleanse it
    user_preds = f.split_user_preds(user_preds, c.NUM_SAMPLES)

    # Remap user predictions as names
    user_preds_names = []
    for char in user_preds:
        try:
            char = int(char)
            user_preds_names.append(c.SENDER_MAPPER_REV[char])
        except:
            user_preds_names.append('err')

    # add score to the cumulative score in the session
    session['cumul_score_user'] += int(score_user)
    session['cumul_score_nn'] += int(score_nn)

    # read out the cumulative score
    cumul_score_user = session['cumul_score_user']
    cumul_score_nn = session['cumul_score_nn']

    # read out the attempts counter
    attempts_counter = session['attempts_counter']

    # read out messages and senders from the session
    df = pd.read_json(session['df'])

    # add user predictions as column to the df
    df['user_pred'] = user_preds_names

    # erase df (messages selection) from session to ensure it gets refreshed
    session.pop('df')

    # render score page
    return render_template(
        'score.html', score_user=score_user, cumul_score_user=cumul_score_user,
        score_nn=score_nn, cumul_score_nn=cumul_score_nn, attempts_counter=attempts_counter, df=df)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

