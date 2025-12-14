from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import json
import datetime
import os
from typing import Dict, List, Any

app = Flask(__name__)
app.secret_key = 'english-test-secret-key-2024'  # Change in production!

# Define English test questions with CEFR levels
ENGLISH_QUESTIONS = [
    {
        'id': 1,
        'question': '"What\'s your name?"',
        'options': [
            "I'm fine, thanks.",
            "I'm 25 years old.", 
            "My name is Sarah.",
            "I live in London."
        ],
        'correct': 2,
        'cefr_level': 'A1',
        'explanation': 'Basic self-introduction question'
    },
    {
        'id': 2,
        'question': '"Would you like ______ coffee?"',
        'options': ["a", "an", "some", "any"],
        'correct': 2,
        'cefr_level': 'A1',
        'explanation': 'Use "some" in polite offers'
    },
    {
        'id': 3,
        'question': 'She ______ to work every day.',
        'options': ["go", "goes", "is going", "went"],
        'correct': 1,
        'cefr_level': 'A1',
        'explanation': 'Present simple for routines'
    },
    {
        'id': 4,
        'question': 'Yesterday, we ______ a great film at the cinema.',
        'options': ["see", "are seeing", "saw", "seen"],
        'correct': 2,
        'cefr_level': 'A2',
        'explanation': 'Past simple for completed actions'
    },
    {
        'id': 5,
        'question': 'This is the restaurant ______ we had our first date.',
        'options': ["which", "where", "who", "when"],
        'correct': 1,
        'cefr_level': 'A2',
        'explanation': 'Relative clause for places'
    },
    {
        'id': 6,
        'question': 'If it rains tomorrow, we ______ the picnic.',
        'options': ["cancel", "will cancel", "would cancel", "cancelled"],
        'correct': 1,
        'cefr_level': 'A2',
        'explanation': 'First conditional for real possibilities'
    },
    {
        'id': 7,
        'question': 'He\'s not here. He ______ left the office already.',
        'options': ["might", "should", "must have", "would have"],
        'correct': 2,
        'cefr_level': 'B1',
        'explanation': 'Modal verbs for deductions about the past'
    },
    {
        'id': 8,
        'question': 'By the time you arrive, I ______ the report.',
        'options': ["finish", "will finish", "will have finished", "am finishing"],
        'correct': 2,
        'cefr_level': 'B1',
        'explanation': 'Future perfect for actions completed before a future time'
    },
    {
        'id': 9,
        'question': 'The project was completed ______ my colleague.',
        'options': ["with", "by", "from", "at"],
        'correct': 1,
        'cefr_level': 'B1',
        'explanation': 'Passive voice with agent'
    },
    {
        'id': 10,
        'question': 'I\'m not used to ______ so early.',
        'options': ["wake up", "waking up", "have woken up", "be waking up"],
        'correct': 1,
        'cefr_level': 'B1',
        'explanation': 'Gerund after "used to"'
    },
    {
        'id': 11,
        'question': 'If I ______ you, I would take that job.',
        'options': ["am", "was", "were", "have been"],
        'correct': 2,
        'cefr_level': 'B2',
        'explanation': 'Second conditional with subjunctive "were"'
    },
    {
        'id': 12,
        'question': 'The company\'s success is ______ to its innovative approach.',
        'options': ["put down", "set up", "looked after", "carried out"],
        'correct': 0,
        'cefr_level': 'B2',
        'explanation': 'Phrasal verb "put down to" meaning "attributed to"'
    },
    {
        'id': 13,
        'question': 'The instructions were ______ complex that nobody understood them.',
        'options': ["such", "so", "too", "very"],
        'correct': 1,
        'cefr_level': 'B2',
        'explanation': '"So...that" structure for result clauses'
    },
    {
        'id': 14,
        'question': 'The manager insisted that everyone ______ on time.',
        'options': ["is", "be", "will be", "must be"],
        'correct': 1,
        'cefr_level': 'B2',
        'explanation': 'Subjunctive after "insist"'
    },
    {
        'id': 15,
        'question': 'Had I known about the traffic, I ______ a different route.',
        'options': ["would take", "will take", "would have taken", "took"],
        'correct': 2,
        'cefr_level': 'B2',
        'explanation': 'Third conditional for hypothetical past situations'
    },
    {
        'id': 16,
        'question': 'The novel is said ______ into over 30 languages.',
        'options': [
            "to have been translated",
            "to be translated", 
            "to translate",
            "it has been translated"
        ],
        'correct': 0,
        'cefr_level': 'C1',
        'explanation': 'Passive infinitive perfect for reported information'
    },
    {
        'id': 17,
        'question': 'He ______ the consequences of his actions.',
        'options': [
            "thoroughly cogitated",
            "gave no thought to",
            "was in a brown study about", 
            "was oblivious to"
        ],
        'correct': 3,
        'cefr_level': 'C1',
        'explanation': 'Advanced vocabulary for unawareness'
    },
    {
        'id': 18,
        'question': 'The negotiations reached a(n) ______ when neither side would compromise.',
        'options': ["impasse", "epiphany", "consensus", "accord"],
        'correct': 0,
        'cefr_level': 'C1',
        'explanation': 'Advanced vocabulary for deadlock situations'
    },
    {
        'id': 19,
        'question': 'Her behavior was a ______ aberration, completely out of character.',
        'options': ["total", "stark", "profound", "gross"],
        'correct': 0,
        'cefr_level': 'C1',
        'explanation': 'Collocation with "aberration"'
    },
    {
        'id': 20,
        'question': 'The government\'s policy has been widely ______ for its short-sightedness.',
        'options': ["lionized", "admonished", "exonerated", "mitigated"],
        'correct': 1,
        'cefr_level': 'C1',
        'explanation': 'Advanced vocabulary for criticism'
    }
]

def calculate_cefr_level(score: int, correct_cefr_levels: List[str]) -> str:
    """Calculate CEFR level based on score and which level questions were answered correctly"""
    
    if score <= 6:
        return "A1-A2 (Beginner to Elementary)"
    elif score <= 10:
        return "B1 (Intermediate)"
    elif score <= 15:
        return "B2 (Upper-Intermediate)"
    elif score <= 19:
        return "C1 (Advanced)"
    else:
        return "C2 (Proficient)"

def get_level_description(level: str) -> str:
    """Get description for CEFR level"""
    descriptions = {
        "A1-A2 (Beginner to Elementary)": "Can understand and use basic everyday expressions",
        "B1 (Intermediate)": "Can handle most situations while traveling in English-speaking areas",
        "B2 (Upper-Intermediate)": "Can interact with native speakers with fluency and spontaneity",
        "C1 (Advanced)": "Can use language flexibly and effectively for social, academic and professional purposes",
        "C2 (Proficient)": "Can understand with ease virtually everything heard or read"
    }
    return descriptions.get(level, "Unknown level")

def save_to_report(user_data: Dict[str, Any]) -> None:
    """Save user test results to report.json"""
    report_file = 'report.json'
    
    if os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            try:
                reports = json.load(f)
            except json.JSONDecodeError:
                reports = {}
    else:
        reports = {}
    
    reports[user_data['user_id']] = user_data
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False, default=str)

@app.route('/')
def index():
    """Home page - user enters ID and starts test"""
    session.clear()
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_test():
    """Start the test - record start time and initialize session"""
    user_id = request.form.get('user_id', '').strip()
    
    if not user_id:
        return redirect(url_for('index'))
    
    session['user_id'] = user_id
    session['start_time'] = datetime.datetime.now().isoformat()
    session['current_question'] = 0
    session['answers'] = []
    
    return redirect(url_for('show_question', question_num=1))

@app.route('/question/<int:question_num>')
def show_question(question_num):
    """Display a specific question"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    if question_num < 1 or question_num > len(ENGLISH_QUESTIONS):
        return redirect(url_for('show_question', question_num=1))
    
    session['current_question'] = question_num - 1
    question = ENGLISH_QUESTIONS[question_num - 1]
    
    return render_template('question.html', 
                         question=question, 
                         question_num=question_num,
                         total_questions=len(ENGLISH_QUESTIONS))

@app.route('/answer', methods=['POST'])
def process_answer():
    """Process user's answer and move to next question or show results"""
    if 'user_id' not in session or 'current_question' not in session:
        return redirect(url_for('index'))
    
    current_q_index = session['current_question']
    selected_answer = request.form.get('answer')
    
    if 'answers' not in session:
        session['answers'] = []
    
    while len(session['answers']) <= current_q_index:
        session['answers'].append(None)
    
    session['answers'][current_q_index] = int(selected_answer) if selected_answer else None
    session.modified = True
    
    next_question = current_q_index + 2
    
    if next_question <= len(ENGLISH_QUESTIONS):
        return redirect(url_for('show_question', question_num=next_question))
    else:
        return redirect(url_for('show_results'))

@app.route('/results')
def show_results():
    """Calculate and display test results"""
    if 'user_id' not in session or 'answers' not in session:
        return redirect(url_for('index'))
    
    score = 0
    user_answers = []
    correct_cefr_levels = []
    
    for i, (question, user_answer) in enumerate(zip(ENGLISH_QUESTIONS, session['answers'])):
        is_correct = user_answer == question['correct']
        if is_correct:
            score += 1
            correct_cefr_levels.append(question['cefr_level'])
        
        user_answers.append({
            'question_id': question['id'],
            'question': question['question'],
            'user_answer': user_answer,
            'correct_answer': question['correct'],
            'is_correct': is_correct,
            'options': question['options'],
            'cefr_level': question['cefr_level'],
            'explanation': question['explanation']
        })
    
    # Calculate level and duration
    level = calculate_cefr_level(score, correct_cefr_levels)
    level_description = get_level_description(level)
    start_time = datetime.datetime.fromisoformat(session['start_time'])
    end_time = datetime.datetime.now()
    duration_seconds = int((end_time - start_time).total_seconds())
    
    # Count correct answers by CEFR level
    level_stats = {}
    for answer in user_answers:
        if answer['is_correct']:
            level_stats[answer['cefr_level']] = level_stats.get(answer['cefr_level'], 0) + 1
    
    results_data = {
        'user_id': session['user_id'],
        'start_time': session['start_time'],
        'end_time': end_time.isoformat(),
        'duration_seconds': duration_seconds,
        'answers': user_answers,
        'score': score,
        'total_questions': len(ENGLISH_QUESTIONS),
        'cefr_level': level,
        'level_description': level_description,
        'level_stats': level_stats
    }
    
    save_to_report(results_data)
    session.clear()
    
    return render_template('results.html', results=results_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)