import requests
import json

from src.code_generation.load_llm import load_llm, get_llm_answer


def read_json_lines_from_url(url):
    """
    Reads JSON Lines data from a given URL and returns a list of Python dictionaries.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    # print(response.text)
    json_data_list = []
    for line in response.iter_lines():
        if line:  # Ensure the line is not empty
            try:
                decoded_line = line.decode('utf-8')
                json_object = json.loads(decoded_line)
                json_data_list.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e} - Line: {line.decode('utf-8')}")
    return json_data_list

def extract_code(text):
    try:
        return text.split("```python")[1].split("```")[0]
    except:
        try:
            return text.split("```")[1].split("```")[0]
        except:
            return text

def find_all_functions(text):
    # Извлекает все функции из текста
    parts = text.split("def ")
    return [p.split("(")[0] for p in parts]

def test_assertions(assertions, assert_fname, test_fname, code):
    # Replace assert_fname on test_fname
    exec_code =  assertions.replace(assert_fname, test_fname)

    # Add assertions to main code
    exec_code =  code + "\n\n" + exec_code
    exec_code = "def run_locally():\n" + exec_code.replace("\n", "\n    ") + "\n"+"run_locally()"
    # Execute code
    try:
        exec(exec_code)
        return True
    except Exception as e:
        return False

def eval_prediction(text, assertions):
    """
    Check if code contains function that solves assertions
    Return True if found at least one function that passed all asserts
    """
    code = extract_code(text)

    # Extract names of functs
    functs = find_all_functions(code)
    assertions = "\n".join(assertions)
    assert_fname = find_all_functions(assertions)[0]

    # Find at least 1 working
    correct_funct = None
    for f in functs:
        if test_assertions(assertions, assert_fname, f, code):
            correct_funct = f
            break
    return correct_funct is not None


if __name__ == "__main__":

    url = "https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/mbpp/mbpp.jsonl"
    data = read_json_lines_from_url(url)

    save_file = "./data_with_prompts.jsonl"
    tokenizer, model = load_llm()


    for ind in range(100):
        ask = {"role": "user", "content": data[ind]['text']}
        res = get_llm_answer([ask], tokenizer, model)
        data[ind]['prediction'] = tokenizer.batch_decode(res)[0]
        if ind % 10 == 9:
            with open(save_file, 'a') as f:
                for item in data[ind-9:ind+1]:
                    json_line = json.dumps(item)
                    f.write(json_line + '\n')

    list_out = []
    for ind in range(60, 100):
        p = False
        try:
            print(ind)
            p = eval_prediction(data[ind]['prediction'], data[ind]['test_list'])

        except:
            p = False
        if p:
            list_out.append(ind)
            print(ind)
    print("Indices of the selected tasks are:", list_out)
