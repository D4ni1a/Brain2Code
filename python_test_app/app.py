from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import json
import datetime
import os
from typing import Dict, List, Any

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production!

# Define the test questions and correct answers
QUESTIONS = [
    {
        'id': 1,
        'question': 'What is the output of `print(2 ** 3)` in Python?',
        'options': ['6', '8', '9', '23'],
        'correct': 1  # index of correct answer (0-based)
    },
    {
        'id': 2,
        'question': 'Which of the following is the correct way to create a list in Python?',
        'options': ['list = (1, 2, 3)', 'list = [1, 2, 3]', 'list = {1, 2, 3}', 'list = <1, 2, 3>'],
        'correct': 1
    },
    {
        'id': 3,
        'question': 'What does the `len()` function do?',
        'options': [
            'Returns the largest element in a list.',
            'Returns the data type of an object.',
            'Returns the number of items in an object.',
            'Converts a value to an integer.'
        ],
        'correct': 2
    },
    {
        'id': 4,
        'question': 'How do you start a `for` loop that iterates 5 times?',
        'options': [
            'for i in range(5):',
            'for i in range(1, 5):',
            'for i in range(0, 5):',
            'Both a and c are correct.'
        ],
        'correct': 3
    },
    {
        'id': 5,
        'question': 'What is the purpose of the `def` keyword?',
        'options': [
            'To define a constant.',
            'To define a class.',
            'To define a function.',
            'To define a variable.'
        ],
        'correct': 2
    },
    {
        'id': 6,
        'question': 'What will be the value of `my_list` after this code runs?\nmy_list = [1, 2, 3]\nmy_list.append([4, 5])',
        'options': [
            '[1, 2, 3, 4, 5]',
            '[1, 2, 3, [4, 5]]',
            '[1, 2, 3, 4, [5]]',
            'An error occurs.'
        ],
        'correct': 1
    },
    {
        'id': 7,
        'question': 'In Object-Oriented Programming (OOP), what is the main purpose of `__init__`?',
        'options': [
            'To terminate an object.',
            'To initialize a newly created object.',
            'To represent the class as a string.',
            'To compare two objects.'
        ],
        'correct': 1
    },
    {
        'id': 8,
        'question': 'What is a key difference between a list and a tuple?',
        'options': [
            'Lists are immutable, tuples are mutable.',
            'Lists are ordered, tuples are unordered.',
            'Lists are mutable, tuples are immutable.',
            'Lists can hold integers, tuples can hold strings.'
        ],
        'correct': 2
    },
    {
        'id': 9,
        'question': 'What does the following list comprehension create?\n[x*2 for x in range(5)]',
        'options': [
            '[0, 1, 2, 3, 4]',
            '[0, 2, 4, 6, 8]',
            '[2, 4, 6, 8, 10]',
            '[1, 2, 4, 8, 16]'
        ],
        'correct': 1
    },
    {
        'id': 10,
        'question': 'What is the output of this code?\ndef func(a, b=2):\n    return a + b\nprint(func(1))',
        'options': ['3', '12', 'Error, missing argument for `b`.', 'None'],
        'correct': 0
    },
    {
        'id': 11,
        'question': 'What concept does this code demonstrate?\ndef outer_func(msg):\n    def inner_func():\n        print(msg)\n    return inner_func\nmy_func = outer_func("Hello")\nmy_func()',
        'options': ['Inheritance', 'Method Overriding', 'Closure', 'Polymorphism'],
        'correct': 2
    },
    {
        'id': 12,
        'question': 'What is the primary use of the `if __name__ == "__main__":` block?',
        'options': [
            'To make a script run faster.',
            'To define the main class of the program.',
            'To ensure code runs only when the script is executed directly, not when imported.',
            'To mark the beginning of a program\'s execution.'
        ],
        'correct': 2
    },
    {
        'id': 13,
        'question': 'What will the following code output?\na = [1, 2, 3]\nb = a\nb.append(4)\nprint(a)',
        'options': ['[1, 2, 3]', '[1, 2, 3, 4]', '[4]', 'An error'],
        'correct': 1
    },
    {
        'id': 14,
        'question': 'Which of these is a correct way to handle an exception in Python?',
        'options': ['try: ... catch: ...', 'try: ... exception: ...', 'try: ... handle: ...', 'try: ... except: ...'],
        'correct': 3
    },
    {
        'id': 15,
        'question': 'What does the `yield` keyword do in a function?',
        'options': [
            'It terminates the function and returns a value.',
            'It turns the function into a generator.',
            'It pauses the function and returns a value, saving its state for the next call.',
            'Both b and c.'
        ],
        'correct': 3
    },
    {
        'id': 16,
        'question': 'What is the difference between `@staticmethod` and `@classmethod`?',
        'options': [
            'A staticmethod takes the class as the first argument, a classmethod does not.',
            'A classmethod takes the class as the first argument (cls), a staticmethod does not.',
            'There is no functional difference.',
            'Staticmethods can be inherited, classmethods cannot.'
        ],
        'correct': 1
    },
    {
        'id': 17,
        'question': 'In the context of the GIL (Global Interpreter Lock), which statement is true?',
        'options': [
            'It allows multiple threads to execute Python bytecode simultaneously.',
            'It prevents multiple native threads from executing Python bytecode at once.',
            'It is a feature that makes Python especially fast for CPU-bound multithreading.',
            'It only exists in Jython and IronPython.'
        ],
        'correct': 1
    },
    {
        'id': 18,
        'question': 'What is the output of this code involving mutable default arguments?\ndef append_to_list(item, my_list=[]):\n    my_list.append(item)\n    return my_list\nprint(append_to_list(1))\nprint(append_to_list(2))',
        'options': ['[1] [2]', '[1] [1, 2]', '[1, 2] [1, 2]', 'An error'],
        'correct': 1
    },
    {
        'id': 19,
        'question': 'What does the `*args` parameter in a function definition allow you to do?',
        'options': [
            'Pass a dictionary of keyword arguments.',
            'Pass a list of arguments.',
            'Pass a variable number of non-keyword (positional) arguments.',
            'Pass a variable number of keyword arguments.'
        ],
        'correct': 2
    },
    {
        'id': 20,
        'question': 'What is Method Resolution Order (MRO) in Python?',
        'options': [
            'The order in which methods are called in a program.',
            'The order in which Python searches for methods in a class hierarchy.',
            'A list of all methods available in a class.',
            'The rule that determines which method to override.'
        ],
        'correct': 1
    },
    {
        'id': 21,
        'question': 'What is the main purpose of a context manager (`with` statement)?',
        'options': [
            'To define a new block of code.',
            'To manage exceptions in a clean way.',
            'To handle the setup and teardown of resources reliably.',
            'To create a temporary variable scope.'
        ],
        'correct': 2
    },
    {
        'id': 22,
        'question': 'What does this metaclass example do?\nclass Meta(type):\n    def __new__(cls, name, bases, dct):\n        dct[\'version\'] = 1.0\n        return super().__new__(cls, name, bases, dct)\nclass MyClass(metaclass=Meta):\n    pass\nprint(MyClass.version)',
        'options': [
            'It creates a new instance of `MyClass` with a `version` attribute.',
            'It dynamically adds a class attribute `version` to `MyClass` during its creation.',
            'It modifies the `__new__` method of `MyClass`.',
            'It causes an error because `version` is not defined in `MyClass`.'
        ],
        'correct': 1
    },
    {
        'id': 23,
        'question': 'What is the key difference between `__str__` and `__repr__`?',
        'options': [
            '`__str__` is for developers, `__repr__` is for end-users.',
            '`__str__` is for informal output, `__repr__` is for official, unambiguous representation.',
            '`__str__` is called by `print()`, `__repr__` is called by the interpreter.',
            'Both b and c.'
        ],
        'correct': 3
    },
    {
        'id': 24,
        'question': 'How can you achieve "private" attributes in a Python class?',
        'options': [
            'By using the `private` keyword.',
            'By using a single underscore `_attr` (convention) or double underscore `__attr` (name mangling).',
            'It is not possible; all attributes are public in Python.',
            'By defining them inside the `__init__` method only.'
        ],
        'correct': 1
    },
    {
        'id': 25,
        'question': 'What does the `asyncio` library primarily help with?',
        'options': [
            'Parallelizing CPU-bound tasks across multiple cores.',
            'Creating graphical user interfaces.',
            'Writing single-threaded concurrent code using async/await.',
            'Managing database connections synchronously.'
        ],
        'correct': 2
    }
]

def calculate_level(score: int) -> str:
    """Calculate proficiency level based on score"""
    if score <= 10:
        return "Junior"
    elif score <= 20:
        return "Middle"
    else:
        return "Senior"

def save_to_report(user_data: Dict[str, Any]) -> None:
    """Save user test results to report.json"""
    report_file = 'report.json'
    
    # Load existing data or create new structure
    if os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            try:
                reports = json.load(f)
            except json.JSONDecodeError:
                reports = {}
    else:
        reports = {}
    
    # Add new report
    reports[user_data['user_id']] = user_data
    
    # Save back to file
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False, default=str)

@app.route('/')
def index():
    """Home page - user enters ID and starts test"""
    session.clear()  # Clear any previous test data
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_test():
    """Start the test - record start time and initialize session"""
    user_id = request.form.get('user_id', '').strip()
    
    if not user_id:
        return redirect(url_for('index'))
    
    # Initialize test session
    session['user_id'] = user_id
    session['start_time'] = datetime.datetime.now().isoformat()
    session['current_question'] = 0
    session['answers'] = []
    
    return redirect(url_for('show_question', question_num=1))

@app.route('/question/<int:question_num>')
def show_question(question_num):
    """Display a specific question"""
    # Check if test has started
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    # Validate question number
    if question_num < 1 or question_num > len(QUESTIONS):
        return redirect(url_for('show_question', question_num=1))
    
    session['current_question'] = question_num - 1  # 0-based index
    question = QUESTIONS[question_num - 1]
    
    return render_template('question.html', 
                         question=question, 
                         question_num=question_num,
                         total_questions=len(QUESTIONS))

@app.route('/answer', methods=['POST'])
def process_answer():
    """Process user's answer and move to next question or show results"""
    if 'user_id' not in session or 'current_question' not in session:
        return redirect(url_for('index'))
    
    current_q_index = session['current_question']
    selected_answer = request.form.get('answer')
    
    # Store the answer
    if 'answers' not in session:
        session['answers'] = []
    
    # Ensure answers list is long enough
    while len(session['answers']) <= current_q_index:
        session['answers'].append(None)
    
    session['answers'][current_q_index] = int(selected_answer) if selected_answer else None
    session.modified = True
    
    # Move to next question or finish test
    next_question = current_q_index + 2  # +1 for next index, +1 for 1-based numbering
    
    if next_question <= len(QUESTIONS):
        return redirect(url_for('show_question', question_num=next_question))
    else:
        return redirect(url_for('show_results'))

@app.route('/results')
def show_results():
    """Calculate and display test results"""
    if 'user_id' not in session or 'answers' not in session:
        return redirect(url_for('index'))
    
    # Calculate score
    score = 0
    user_answers = []
    
    for i, (question, user_answer) in enumerate(zip(QUESTIONS, session['answers'])):
        is_correct = user_answer == question['correct']
        if is_correct:
            score += 1
        
        user_answers.append({
            'question_id': question['id'],
            'question': question['question'],
            'user_answer': user_answer,
            'correct_answer': question['correct'],
            'is_correct': is_correct,
            'options': question['options']
        })
    
    # Calculate level and duration
    level = calculate_level(score)
    start_time = datetime.datetime.fromisoformat(session['start_time'])
    end_time = datetime.datetime.now()
    duration_seconds = int((end_time - start_time).total_seconds())
    
    # Prepare results data
    results_data = {
        'user_id': session['user_id'],
        'start_time': session['start_time'],
        'end_time': end_time.isoformat(),
        'duration_seconds': duration_seconds,
        'answers': user_answers,
        'score': score,
        'total_questions': len(QUESTIONS),
        'level': level
    }
    
    # Save to report file
    save_to_report(results_data)
    
    # Clear session
    session.clear()
    
    return render_template('results.html', results=results_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)