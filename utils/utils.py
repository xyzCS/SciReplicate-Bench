import os
from openai import OpenAI
import anthropic
import textwrap
import time
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
)
import shutil
import re
from transformers import StoppingCriteria
import Levenshtein
import subprocess
import ast
from typing import Tuple, Union, Dict

def read_python_file(file_path):
    """
    Read a Python file and return its contents as a string.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        str: Contents of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def FormatCheckTool(code):
    # formatted_code = autopep8.fix_code(code)
    formatted_code = textwrap.dedent(code)
    return formatted_code

class MultiTokenStopCriteria(StoppingCriteria):
    def __init__(self, stop_sequences):
        super().__init__()
        self.stop_sequences = stop_sequences  # 形如 [[id1, id2, id3], ...]

    def __call__(self, input_ids, scores, **kwargs):
        for stop_seq in self.stop_sequences:
            if input_ids[0, -len(stop_seq):].tolist() == stop_seq:
                return True
        return False

def clean_generated_output(output: str, stop_words: list) -> str:
    output = output.strip()
    for stop_word in stop_words:
        if output.endswith(stop_word):
            output = output[:-len(stop_word)].rstrip()
    return output

def llm(prompt, stop=["\n"], model='gpt-4o', effort='medium', generate_code=False, local_model=None, local_tokenizer=None):
    if model == 'gpt-4o' or model == 'gpt-4o-mini' or model == 'o1-mini' or model == 'o1' or model == 'o3-mini':
        client = OpenAI(
            api_key = os.environ["OPENAI_API_KEY"]
        ) 
    elif model == 'deepseek-r1' or model == 'deepseek-v3':
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    elif model == 'claude-3-7':
        client = anthropic.Anthropic(
            api_key=os.environ["CLAUDE_API_KEY"],
        )
    elif model == 'gemini-2.0-flash' or model == 'gemini-2.5-pro' or model == 'gemini-2.5-flash' or model == 'gemini-2.0-flash-thinking':
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    while True:
        try:
            if model == 'gpt-4o' or model == 'gpt-4o-mini':
                if generate_code:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful code generation assistant."},
                            {"role": "user", "content": prompt}],
                        temperature=0.5,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    content = response.choices[0].message.content
                    break
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop
                    )
                    content = response.choices[0].message.content
                    break
            elif model == 'o1' or model == 'o1-mini' or model == 'o3-mini':
                if generate_code:
                    response = client.chat.completions.create(
                        model=model,
                        reasoning_effort=effort,
                        messages=[
                            {"role": "system", "content": "You are a helpful code generation assistant."},
                            {"role": "user", "content": prompt}],
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    content = response.choices[0].message.content
                    break
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        reasoning_effort=effort,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop
                    )
                    content = response.choices[0].message.content
                    break
            elif model == 'deepseek-r1':
                model = 'deepseek-reasoner'
                if generate_code:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful code generation assistant."},
                            {"role": "user", "content": prompt}],
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    thinking = response.choices[0].message.reasoning_content
                    after_think = response.choices[0].message.content
                    content = (thinking, after_think)
                    break
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop
                    )
                    thinking = response.choices[0].message.reasoning_content
                    after_think = response.choices[0].message.content
                    content = (thinking, after_think)
                    break
            elif model == 'deepseek-v3':
                model = 'deepseek-chat'
                if generate_code:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful code generation assistant."},
                            {"role": "user", "content": prompt}],
                        temperature=0.5,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    content = response.choices[0].message.content
                    break
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop
                    )
                content = response.choices[0].message.content
                break
            elif model == 'claude-3-7':
                if  stop == ["\n"]:
                    stop = None
                
                message = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=8000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stop_sequences=stop
                )
                content = message.content[0].text
                break
            elif model == 'claude-3-7-think':
                if  stop == ["\n"]:
                    stop = None
                message = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=8000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 16000
                    },
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stop_sequences=stop
                )
                content = message.content[0].text
                break
            elif model == 'gemini-2.0-flash':
                response = client.models.generate_content(
                    model='gemini-2.0-flash', 
                    contents=prompt,
                    config=GenerateContentConfig(
                        safety_settings=safety_settings,
                    ),
                )
                content = response.text
                if not generate_code:
                    index = content.find("Observation:")
                    if index != -1:
                        content = content[:index]
                break
            elif model == 'gemini-2.0-flash-thinking':
                response = client.models.generate_content(
                    model='gemini-2.0-flash-thinking-exp', 
                    contents=prompt,
                    config=GenerateContentConfig(
                        safety_settings=safety_settings,
                    ),
                )
                content = response.text
                if not generate_code:
                    index = content.find("Observation:")
                    if index != -1:
                        content = content[:index]
                break
        except Exception as e:
            print(f"Error: {str(e)}; Waiting for 1 miniute and retrying...")
            time.sleep(60)

    return content

def copy_file(source_path, destination_path):
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Copy the file, preserving metadata
        shutil.copy2(source_path, destination_path)
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

def delete_file(file_path):
    try:
        # Check if file exists
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        else:
            print(f"File {file_path} does not exist")
            return False
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False

def are_strings_same(s1, s2, threshold=0.1):
    """
    Returns True if the normalized Levenshtein distance between s1 and s2 is below
    or equal to the threshold, along with the computed distance and normalized value.
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return True
    distance = Levenshtein.distance(s1, s2)
    normalized_distance = distance / max_len
    return normalized_distance <= threshold

def Is_Exist(Already_Exist, result):
    for i in range(len(Already_Exist)):
        if result.lower() in Already_Exist[i].lower():
            return True
        if are_strings_same(Already_Exist[i], result, threshold=0.1):
            return True
    return False

def extract_pythonfile(action: str) -> str:
    """
    Extract the label from a SearchSection action.
    
    This function supports actions in the following formats:
        - SearchSection[a]
        - SearchSection["a"]
    """
    pattern = r'^SearchFile\[\s*"?([^"\]]+)"?\s*\]$'
    match = re.match(pattern, action)
    if match:
        return match.group(1).strip()
    return None

def extract_code_item(action: str) -> str:
    """
    Extract the label from a SearchSection action.
    
    This function supports actions in the following formats:
        - SearchSection[a]
        - SearchSection["a"]
    """
    pattern = r'^SearchCodeItem\[\s*"?([^"\]]+)"?\s*\]$'
    match = re.match(pattern, action)
    if match:
        return match.group(1).strip()
    return None

def extract_label_web(action: str) -> str:
    """
    Extract the label from a SearchSection action.
    
    This function supports actions in the following formats:
        - SearchSection[a]
        - SearchSection["a"]
    """
    pattern = r'^SearchWeb\[\s*"?([^"\]]+)"?\s*\]$'
    match = re.match(pattern, action)
    if match:
        return match.group(1).strip()
    return None

def extract_label_paper(action: str) -> str:
    """
    Extract the label from a SearchPaper action.
    
    This function supports actions in the following formats:
        - SearchPaper[query]
        - SearchPaper["query"]
    """
    # 更健壮的正则表达式，处理引号内可能包含的任何字符
    pattern = r'^SearchPaper\[\s*"(.*?)"\s*\]$'
    match = re.match(pattern, action)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配不带引号的情况
    pattern2 = r'^SearchPaper\[\s*([^\]]+)\s*\]$'
    match = re.match(pattern2, action)
    if match:
        return match.group(1).strip()
    
    return None

def recover_1(data, Benchmark_path_global):
    repo_path = data['project_path']
    Benchmark_path = os.path.join(Benchmark_path_global, repo_path)
    repo_path = data['project_path']
    CodeRepoPath = Benchmark_path
    for i in range(0, len(data['task_details'])):
        task = data['task_details'][i]
        file_path = os.path.join(CodeRepoPath, task['completion_path'][2:])
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(task['ori_python_file'])

def Extract_Function_Definition(task, Python_File, dependency=False, external_APIs=False, missing_details=False):
    Python_File = Python_File.split('\n')
    function_definition = '\n'.join(Python_File[task['signature_position'][0]-1:task['signature_position'][1]])
    Missing_details = task['Missing_Mismatched_details']['string']
    Arguements = task['Arguments']['string']
    Dependencies = task['dependency']['string']
    External_APIs = task['external_APIs']['string']
    Return = task['Return']['Return_String']
    Function_Comment = "\"\"\""
    Function_Comment = Function_Comment + "\n" + Arguements + Return
    if missing_details:
        Function_Comment = Function_Comment + Missing_details
    if dependency:
        Function_Comment = Function_Comment + Dependencies
    if external_APIs:
        Function_Comment = Function_Comment + External_APIs 
    Function_Comment += "\"\"\""
    Function_Comment  = textwrap.indent(Function_Comment, '    ' * task['indent'])
    Function_string = function_definition + '\n' + Function_Comment
    return Function_string

def recover(data, task_id, Benchmark_path_global):
    repo_path = data['project_path']
    Benchmark_path = os.path.join(Benchmark_path_global, repo_path)
    repo_path = data['project_path']
    CodeRepoPath = Benchmark_path
    for i in range(0, len(data['task_details'])):
        task = data['task_details'][i]
        file_path = os.path.join(CodeRepoPath, task['completion_path'][2:])
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(task['ori_python_file'])
    
    task = data['task_details'][task_id]
    file_path = os.path.join(CodeRepoPath, task['completion_path'][2:])
    Python_File = task['ori_python_file'].split('\n')
    Function_string = Extract_Function_Definition(task, task['ori_python_file'])
    Python_File[task['signature_position'][0]-1:task['body_position'][1]] = Function_string.split('\n')
    Python_File = '\n'.join(Python_File)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(Python_File)

def remove_space(code_list):
    for m in range(len(code_list)-1, -1, -1):
        if code_list[m] == '':
            code_list.pop(m)
        else:
            code_list[m] = code_list[m].replace('\n', '')
    return code_list

def extract_error_lines(stderr_str):
    """
    Return only lines containing typical Python exception messages
    or the traceback lines themselves.
    """
    # Some possible keywords to keep (you can customize):
    patterns = [
        r'^Traceback \(most recent call last\):',
        r'^\s*File\s+\"',
        r'^(\w+Error|Exception):',
        r'^.*Run:.*$',
        r'^.*Error:.*$',  # added to catch error lines that don't match the others
    ]

    keep_regex = re.compile('|'.join(patterns))
    # We'll collect lines that match any of the above patterns
    keep_lines = []

    for line in stderr_str.splitlines():
        if keep_regex.search(line):
            keep_lines.append(line)
        # Also keep lines following a traceback (heuristic: if the line is indented after a 'File ' line)
        elif keep_lines and line.startswith('    '):
            # It's likely part of the traceback stack
            keep_lines.append(line)

    # If we found anything, return it, otherwise return entire stderr
    return "\n".join(keep_lines) if keep_lines else stderr_str

def run_compare(CodeRepoPath: str,  file_name: str, command: str, repo_path, conda_env, gpu_id, conda_env_path) -> Tuple[bool, Union[str, Exception]]:
    """
    Executes a Python script in the specified Anaconda environment and GPU context.

    :param file_name: The Python script file to run.
    :param function_call: The function call to execute in the script.
    :param hyperparams: A dictionary of hyperparameters to provide to the script.
    :return: A tuple with success flag and output or error message.
    """
    conda_env = os.path.join(conda_env_path, conda_env)
    file_path = os.path.join(repo_path, file_name)
    
    if not os.path.exists(file_path):
        return False, f"File '{file_name}' does not exist in the repository."

    try:
        # Prepare the command to activate conda and run the script with the assigned GPU
        command = (
            f"cd {CodeRepoPath} && "
            f"/bin/bash -c '"
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {conda_env} && "
            f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
            f"{command}'"
        )

        # Execute the command
        result = subprocess.run(command, 
                                shell=True, 
                                text=True, 
                                stderr=subprocess.PIPE,  # Capture only stderr
                                stdout=subprocess.DEVNULL,  # Suppress all stdout
                                cwd=repo_path,
                                timeout=60*30
                                )

        if result.returncode == 0:
            return True, ""
        else:
            # Now filter out only the traceback portion from stderr
            return False, extract_error_lines(result.stderr)
    except Exception as e:
        if isinstance(e, subprocess.TimeoutExpired):
            return False, "Execution timed out after 30 minutes. Far too long!"
        return False, str(e)

def get_return_variable_names(return_line):
    """
    Given a single line of code that contains a return statement,
    this function returns a list of variable names that are being returned.
    
    Examples:
        "return sum_counts" -> ["sum_counts"]
        "return sum_counts, sub_counts" -> ["sum_counts", "sub_counts"]
    
    Parameters:
        return_line (str): A line of Python code containing a return statement.
    
    Returns:
        list of str: The names of the variables in the return expression.
                     If the return expression is not simply a list/tuple of names,
                     then only the names that can be extracted will be returned.
    """
    # Remove any leading/trailing whitespace.
    line = return_line.strip()
    
    # Check that the line starts with "return"
    if not line.startswith("return"):
        return []
    
    # Get the part after the "return" keyword.
    expr_text = line[len("return"):].strip()
    
    # If nothing is returned, return an empty list.
    if not expr_text:
        return []
    
    try:
        # Parse the expression using ast. We use mode="eval" because
        # the expression is expected to be a single expression (not a full statement).
        expr_node = ast.parse(expr_text, mode="eval").body
    except Exception as e:
        print("Error parsing the return expression:", e)
        return []
    
    names = []
    
    # If the return expression is a tuple, extract each element.
    if isinstance(expr_node, ast.Tuple):
        for elt in expr_node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                # If it's not a simple variable name, try to get its source code.
                names.append(ast.unparse(elt) if hasattr(ast, "unparse") else "<complex_expression>")
    elif isinstance(expr_node, ast.Name):
        names.append(expr_node.id)
    else:
        # For other kinds of expressions (e.g. a dictionary or function call),
        # attempt to extract names if possible.
        # This might be a single complex expression.
        try:
            # If running Python 3.9+, ast.unparse() is available.
            expression = ast.unparse(expr_node) if hasattr(ast, "unparse") else expr_text
            names.append(expression)
        except Exception:
            names.append("<unknown>")
    
    return names

def return_test_code(task_id, variables, output_dir):
    variable_str = "task1 = ["
    for i in range(len(variables)):
        variable_str += f'{variables[i]},'
    variable_str = variable_str[:-1] + "]\n"
    test_code = f"""
import pickle, os
outputDirTask1 = '{output_dir}/task{str(task_id+1)}'
{variable_str}
if os.path.exists(outputDirTask1) == False:
    os.makedirs(outputDirTask1)
files = [f for f in os.listdir(outputDirTask1) if os.path.isfile(os.path.join(outputDirTask1, f))]
num_file = len(files)
if num_file < 10:
    id = num_file
    if os.path.exists(outputDirTask1) == False:
        os.makedirs(outputDirTask1)
    filepathTask = os.path.join(outputDirTask1, 'output_' + str(id) + '.pickle')
    filepathTask = open(filepathTask, "wb")
    pickle.dump(task1, filepathTask)
    """
    return test_code.split('\n')

def insert_code_before_returns(code_A, task_id, output_dir):
    """
    For each line in code_A that starts with 'return' (ignoring whitespace),
    insert code_B (with adjusted indentation) immediately before that line.
    
    Args:
        code_A (list of str): Original code as a list of lines.
        task_id (int): Task identifier used to generate output file paths.
        output_dir (str): Directory to save output files.
    
    Returns:
        list of str: New code_A with code_B inserted before every return.
    """
    base_indent_B = 0
    new_code = []

    # Regex patterns
    return_regex = re.compile(r'^\s*return\b')
    # Match only the beginning of a function definition (without requiring the colon)
    def_start_pattern = re.compile(r'^\s*def\s+([a-zA-Z_]\w*)\s*\(')
    comma = re.compile(r'\)(.*?):')
    # Dictionary to keep track of function definitions: {line_num: (start_line, end_line)}
    function_defs = {}
    
    # First pass: identify all function definitions
    i = 0
    while i < len(code_A):
        line = code_A[i]
        # If this line starts a function definition
        if def_start_pattern.search(line):
            func_start = i
            # Look for the end of the function signature (line with a colon)
            j = i
            while j < len(code_A) and not comma.search(code_A[j]):
                j += 1
            
            if j < len(code_A):  # Found the closing colon
                func_end = j
                # Store this function definition
                function_defs[func_start] = (func_start, func_end)
                if i == j:
                    i += 1
                else:
                    i = j  # Skip to the end of the signature
            else:
                i += 1  # No closing colon found, move to next line
        else:
            i += 1
    
    # Find the first function definition (main function for the task)
    begin_def_line = min(function_defs.keys()) if function_defs else 0
    last_def_list = []
    
    # Second pass: process return statements and insert test code
    for i in range(len(code_A)):
        line = code_A[i]
        if return_regex.search(line):
            # Find which function this return belongs to
            last_def = None
            for start_line in sorted(function_defs.keys(), reverse=True):
                if start_line < i and start_line not in last_def_list:
                    last_def = start_line
                    break
            
            # Skip if this return is not in the main function and we haven't processed it yet
            last_return = False
            for m in range(i+1, len(code_A)):
                if return_regex.search(code_A[m]):
                    last_return = True
                    break
            
            if last_def != begin_def_line and last_def not in last_def_list and last_return:
                last_def_list.append(last_def)
                new_code.append(line)
                continue
                
            # Extract return variables and generate test code
            return_string = ""
            for m in range(i, len(code_A)):
                return_string += code_A[m]
                try:
                    ast.parse(return_string.lstrip())
                    break
                except Exception as e:  
                    print("")
                
            variables = get_return_variable_names(return_string)
            code_B = return_test_code(task_id, variables, output_dir)
            return_indent = line[:len(line) - len(line.lstrip())]
            
            # Insert the test code with adjusted indentation
            for b_line in code_B:
                b_line_strip = b_line.lstrip('\n')
                # For blank lines, just add a newline with the return_indent
                if b_line_strip.strip() == "":
                    new_line = return_indent + b_line_strip
                else:
                    # Calculate proper indentation
                    current_indent = len(b_line) - len(b_line.lstrip())
                    # Preserve relative indentation structure
                    if current_indent >= base_indent_B:
                        extra_indent = b_line[base_indent_B:current_indent]
                    else:
                        extra_indent = b_line[:current_indent]
                    new_line = return_indent + extra_indent + b_line.lstrip()
                new_code.append(new_line)
        
        # Always add the current line
        new_code.append(line)
    # print('\n'.join(new_code))
    return new_code

def read_python_file(file_path):
    """
    Read a Python file and return its contents as a string.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        str: Contents of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def remove_comments_from_function(function_code):
    """
    Removes all comments from a given Python function code, preserving indentation and
    removing extra blank lines.

    Args:
        function_code (str): The code of the Python function as a string.

    Returns:
        str: The function code with comments removed and extra blank lines reduced.
    """
    # Remove inline comments
    function_code = re.sub(r'#[^\n]*', '', function_code)
    # Remove multiline comments (docstrings or multiline strings)
    function_code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', function_code, flags=re.DOTALL)
    # Split code into lines
    lines = function_code.splitlines()
    # Remove lines that are only whitespace and rstrip to preserve indent but remove trailing spaces
    cleaned_lines = [line.rstrip() for line in lines if line.strip() != '']
    return "\n".join(cleaned_lines)










