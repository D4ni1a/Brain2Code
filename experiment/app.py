from flask import Flask, render_template, jsonify, request
import random
import time
import datetime
import re
import os

app = Flask(__name__)

# Global configuration variables
# REST_TIME = 3  # seconds
# GET_READY_SIGNAL_TIME = 0.25  # seconds
FONT_FAMILY = "Verdana, sans-serif"
FONT_SIZE = "22px"
CROSS_SIZE = "20px"  # размер крестика
READ_TIME = 1  # seconds to read a word
REST_TIME_WORD = 2  # seconds for blank screen between words
GET_READY_SIGNAL_TIME_WORD = 1  # seconds before switch to play sound between words
REST_TIME_SENT = 3  # seconds for blank screen between sentences
GET_READY_SIGNAL_TIME_SENT = 1  # seconds before switch to play sound between sentences
REST_TIME_SESSION = 5  # seconds for blank screen between sessions
GET_READY_SIGNAL_TIME_SESSION = 1  # seconds before switch to play sound between sessions
BACKGROUND_COLOR = "#f8f4f0"
SENTENCES_PATH = "code_tasks.txt"
ORDER_PATH = "tasks_order.txt"
SENTENCES_NUM = 15
SENTENCES_IN_SESSION = 5

# Global state
sentences = []
word_list = []
current_index = 0
experiment_started = False
experiment_finished = False


def load_and_prepare_sentences():
    """Load sentences from file, process them, and save order"""
    global sentences, word_list

    try:
        # Read sentences from file
        with open(SENTENCES_PATH, 'r', encoding='utf-8') as f:
            raw_sentences = [line.strip() for line in f if line.strip()]

        # Process sentences: lowercase, remove punctuation, split into words
        processed_sentences = []
        for sentence in raw_sentences:
            # Remove punctuation except apostrophes and hyphens within words
            cleaned = re.sub(r'[^\w\s\'-]', '', sentence.lower())
            words = cleaned.split()
            processed_sentences.append(words)

        # Shuffle and select first SENTENCES_NUM sentences
        if len(processed_sentences) < SENTENCES_NUM:
            print(f"Warning: Only {len(processed_sentences)} sentences found, using all available")
            selected_indices = list(range(len(processed_sentences)))
            selected_sentences = processed_sentences
        else:
            # Create indices and shuffle
            indices = list(range(len(processed_sentences)))
            random.shuffle(indices)
            selected_indices = indices[:SENTENCES_NUM]
            selected_sentences = [processed_sentences[i] for i in selected_indices]

        sentences = selected_sentences
        print(sentences)

        # Save order to ORDER_PATH
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ORDER_PATH, 'a', encoding='utf-8') as f:
            f.write(f"\n{timestamp}: {','.join(map(str, selected_indices))}\n")

        # Create flat word list with sentence boundaries
        word_list = []
        for sentence_words in sentences:
            word_list.extend(sentence_words)
            word_list.append(None)  # None represents sentence boundary
        print("word_list\n",word_list)

        print(f"Loaded {len(sentences)} sentences")
        print(f"Total words (including boundaries): {len(word_list)}")

    except Exception as e:
        print(f"Error loading sentences: {e}")
        # Fallback to example sentences
        sentences = [
            ["find", "the", "perimeter", "of", "a", "square"],
            ["find", "similar", "elements", "from", "the", "given", "two", "tuple", "lists"],
            ["find", "squares", "of", "individual", "elements", "in", "a", "list", "using", "lambda", "function"]
        ]
        word_list = []
        for sentence_words in sentences:
            word_list.extend(sentence_words)
            word_list.append(None)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html',
                           background_color=BACKGROUND_COLOR,
                           font_family=FONT_FAMILY,
                           font_size=FONT_SIZE,
                           cross_size=CROSS_SIZE)


@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    """Start the experiment"""
    global experiment_started
    experiment_started = True
    return jsonify({'status': 'started'})


@app.route('/get_next_word', methods=['GET'])
def get_next_word():
    """Get the next word and timing information"""
    global current_index, experiment_finished, word_list

    if not experiment_started:
        return jsonify({'status': 'not_started'})

    if current_index >= len(word_list):
        experiment_finished = True
        return jsonify({'status': 'finished'})

    # Calculate timing based on position
    word = word_list[current_index]
    print("started")
    print(current_index, word)
    print(word_list)

    # Determine if it's a word or sentence boundary
    if word is None:
        # Sentence boundary
        sentence_num = current_index // (max(len(s) for s in sentences) + 1)
        session_num = sentence_num // SENTENCES_IN_SESSION

        if sentence_num % SENTENCES_IN_SESSION == SENTENCES_IN_SESSION - 1 or sentence_num == len(sentences) - 1:
            # End of session or end of experiment
            rest_time = REST_TIME_SESSION
            signal_time = GET_READY_SIGNAL_TIME_SESSION
        else:
            # Between sentences
            rest_time = REST_TIME_SENT
            signal_time = GET_READY_SIGNAL_TIME_SENT

        current_index += 1
        return jsonify({
            'status': 'boundary',
            'word': '',
            'rest_time': rest_time,
            'signal_time': signal_time,
            'is_end_of_session': sentence_num % SENTENCES_IN_SESSION == SENTENCES_IN_SESSION - 1
        })
    else:
        # Regular word
        sentence_num = 0
        word_count = 0
        for i, sentence in enumerate(sentences):
            if current_index < word_count + len(sentence):
                sentence_num = i
                break
            word_count += len(sentence) + 1  # +1 for boundary

        # Check if next item is boundary
        is_last_word_in_sentence = current_index + 1 >= len(word_list) or word_list[current_index + 1] is None

        rest_time = REST_TIME_WORD
        signal_time = GET_READY_SIGNAL_TIME_WORD

        response = {
            'status': 'word',
            'word': word,
            'rest_time': rest_time,
            'signal_time': signal_time,
            'read_time': READ_TIME,
            'sentence_num': sentence_num + 1,
            'word_num': current_index - word_count + 1,
            'is_last_word_in_sentence': is_last_word_in_sentence
        }

        current_index += 1
        return jsonify(response)


@app.route('/check_status', methods=['GET'])
def check_status():
    """Check experiment status"""
    return jsonify({
        'started': experiment_started,
        'finished': experiment_finished,
        'current_index': current_index,
        'total_words': len(word_list)
    })


@app.route('/reset', methods=['POST'])
def reset_experiment():
    """Reset experiment state (for debugging/development)"""
    global current_index, experiment_started, experiment_finished
    current_index = 0
    experiment_started = False
    experiment_finished = False
    return jsonify({'status': 'reset'})


# Initialize sentences on startup
load_and_prepare_sentences()

if __name__ == '__main__':
    # input()
    app.run(debug=True, port=5000)
    # input()