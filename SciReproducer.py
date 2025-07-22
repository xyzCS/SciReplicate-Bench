import os
from utils.CodeAgentTools import analyze_file, run_code
from utils.WebSearch import WebSearch
from utils.PaperAgentTools import extract_section, Download_Latex_Repo, read_bib_files, Extract_Latex_Section, clean_latex_content
from utils.utils import Is_Exist, recover
import pickle
import json
import shutil
from utils.utils import llm, extract_pythonfile, extract_code_item, extract_label_web, extract_label_paper, Extract_Function_Definition
import argparse
import re
from pathlib import Path
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

CORRECT = """

Response Template:
    Thought: I think ...
    Action: SearchWeb[Query] or SearchPaper[query] or SearchSection[x] or SearchLiterature[key, query] or SearchFile[M] or SearchCodeItem[M] or GenerateCode
    Observation: Outcome of the action.

IMPORTANT: If you prepare to generate the code, make sure to generate the action "Action: GenerateCode" !!!!

Your answer is:

"""

CORRECT_PaperAgent = """

Response Template:
    Thought: I think ...
    Action: SearchPaper[query] or SearchSection[x] or SearchLiterature[key, query] or Finish
    Observation: Outcome of the action.

Your answer is:

"""

GENCODE = r"""
Please complete the target function (or method) and provide the pure code output without any additional text. Include comments in the code following these specific guidelines:

Code Comments:
    1. Focus on Reasoning: Comments should explain the reasoning behind the code generation process, as derived from the LaTeX description.
    2. No Implementation Details: Avoid including any code-specific implementation details in the comments.
    3. Mapping to LaTeX: Each comment must indicate which functionality described in the LaTeX code is implemented in the subsequent Python code snippet.
    4. No overlap: The code snippets corresponding to each comment must not overlap, and nesting is not allowed.
    5. Single Function: Implement the code within a single function (or method), without breaking it into multiple functions.
    6. Import Package: Import all packages within the function (or method) to ensure the code is self-contained.
    7. Format, the comment for each snippet should be in the following format:
        # ---------------------------------------------------------------------------
        # Snippet x: Comment here
        # ---------------------------------------------------------------------------
        # [Begin Snippet X]
        Code snippet here
        # [End Snippet X]

[Example]:
```python
def apply_token_level_transformation(
    token_representation,
    auxiliary_representation,
    transformation_indices,
    batch_mask,
    scaling_factor=0.01
):
    import torch
    import numpy as np
    transformed_tokens = token_representation.clone()

    for i, (start_idx, end_idx) in enumerate(transformation_indices):
        # -----------------------------------------------------------------------
        # Snippet 1: We first verify if the current batch item is valid by checking the
        # batch_mask, analogous to referencing an in-context example \(\mathbf{S}\)
        # that has a corresponding reference \(\mathbf{H_r}\) in the LaTeX snippet.
        # -----------------------------------------------------------------------
        # [Begin Snippet 1]
        if batch_mask[i]:
        # [End Snippet 1]
        
            # -------------------------------------------------------------------
            # Snippet 2: Here, we apply a basic shift to the token_representation by
            # incorporating a slice from the auxiliary_representation, akin to
            # combining \(\mathbf{H}\) with a portion of \(\mathbf{H_r}\) for
            # enhanced alignment.
            # -------------------------------------------------------------------
            # [Begin Snippet 2]
            segment_main = transformed_tokens[i, start_idx:end_idx, :]
            segment_aux = auxiliary_representation[i, start_idx:end_idx, :]
            # [End Snippet 2]

            # -------------------------------------------------------------------
            # Snippet 3: The transformation is a simple element-wise operation combined with
            # the scaling_factor, illustrating a simplified version of the
            # alignment concept from the LaTeX, which might involve more complex
            # contrastive or attentional calculations.
            # -------------------------------------------------------------------
            # [Begin Snippet 3]
            updated_segment = torch.add(segment_main, torch.mul(segment_aux, scaling_factor))
            transformed_tokens[i, start_idx:end_idx, :] = updated_segment
            # [End Snippet 3]

    # ---------------------------------------------------------------------------
    # Snippet 4: The final transformed_tokens are now partially aligned with the auxiliary
    # reference, reflecting the notion of augmenting token-level outputs
    # (Equation references in the LaTeX snippet would correspond to eq. (2-3) or
    # similar definitions of reference alignment).
    # ---------------------------------------------------------------------------
    # [Begin Snippet 4]
    return transformed_tokens
    # [End Snippet 4]
```
Your answer:
"""

AUGMENTLATEX = r"""
Write a comprehensive report of the extracted process for the next stage of code generation.

Requirements:
1. Focus on the most critical information that will directly support code reproduction.
2. make the report concise and to the point.
3. When information from SearchLiterature action conflicts with the current paper, always prioritize the information from the current paper.
4. Structure the report as a series of information points using the following format:

```report

Information Point 1: [Brief descriptive title]
Overview: [Concise description of what this information pertains to]
Source: [Where the information comes from, select from "Target Paper", "Relevant Literature"]
Extracted Information:
    *[Key detail 1 with precise values, parameters, or steps where applicable]
    *[Key detail 2 with precise values, parameters, or steps where applicable]

Information Point 2: [Brief descriptive title]
Overview: [Concise description of what this information pertains to]
Action: [Where the information comes from, select from "Target Paper", "Relevant Literature]
Extracted Information:
    *[Key detail 1]
    *[Key detail 2]

```

[The Report]


"""

def remove_figures(latex_text):
    """
    Removes all figure and table environments from the LaTeX string.
    
    This function searches for blocks starting with \begin{figure} or \begin{table}
    (with possible options in square brackets) and removes everything up to the 
    corresponding \end{figure} or \end{table}.
    """
    # Pattern to match \begin{figure}[optional options] ... \end{figure} and similarly for table.
    pattern = r"\\begin\{(figure\*?|table\*?)\}(?:\[[^\]]*\])?.*?\\end\{\1\}"
    # Use DOTALL flag so that newline characters are included.
    cleaned_text = re.sub(pattern, "", latex_text, flags=re.DOTALL)
    return cleaned_text

def judge_repeat_action_PaperAgent(action_history, current_action, model_name, local_model=None, local_tokenizer=None):
    prompt=r"""
I will provide the description of three search actions, a search action history and a new search action, you should determine whether the new search action has been taken before. 

[Description of the search actions]

1. SearchPaper[query]
    Description: When a variable, concept, or any other element appears in the target section without its full definition or sufficient details, use this action to search for the complete information in the full paper.
    Parameters:
        - 'query' (string): A query describing the information that needs to be located within the full paper.
    Examples:
        - If the LaTeX contains: "We use the concept of $X_i$ to define Y," then the action should be: SearchPaper["The definition of $X_i$"]
        - If the LaTeX contains: "The function $f(x)$ is defined based on the properties of $\mathcal{G}$.", then the action should be: SearchPaper["The properties of $\mathcal{G}$"]

2. SearchSection[x]
    Description: If the target section references another section in the paper with the title x, extract the information from the referenced section and return SearchSection[x].
    Parameters: 'x' (string): The title of the referenced section.
    Example: 
        - Latex: "The full derivation of our loss function can be found in method Section .", Action: SearchSection["method"]

3. SearchLiterature[key, query]
    Description: If the target section cites another paper (\cite{label}) and you determine that some information needs to be retrieved from that paper, return SearchLiterature[label, query], where query is the specific information you need to look for in the referenced paper.
    Parameters: 
        - 'key' (string): The citation key of the referenced paper. In LaTeX, when citing a paper, we use \cite{x}, where x represents the citation key.
        - 'query' (string): The specific information to search for in the referenced paper.
    Example: I
        - Latex: "We adopt the metric proposed in \cite{wang2025}". Action: SearchLiterature["wang2025", "The proposed metric in the paper"]
        - Latex: "The algorithm is based on the work of \cite{smith2018}". Action: SearchLiterature["smith2018", "The algorithm details in the paper"]
        - Latex: "The dataset is based on the study by \cite{jones2020}". Action: SearchLiterature["jones2020", "The dataset details in the paper"]

"""
    prompt += f"""
[Search Action History]

{action_history}

[Currrent Search Action]

{current_action}

Please check if the current search action has already been performed by referring to the search history. Follow these criteria:
1. If the current search action has been performed before, return True; otherwise, return False.
2. For SearchPaper and SearchLiterature actions, consider the current action repeated if any history action serves the same function (even if not character-by-character identical) as the current search.
3. For SearchSection actions, consider the current action repeated if the section title is character-by-character identical to a previous search.
4. Provide only True or False as your answer, with no additional comments or explanations.

[Your Answer]
""" 
    result_tmp = llm(prompt=prompt, model=model_name, generate_code=False, local_model=local_model, local_tokenizer=local_tokenizer)
    if 'R1' in model_name:
        result_tmp = result_tmp[1]
    return result_tmp

def judge_repeat_action_CodeAgent(action_history, current_action, model_name, local_model=None, local_tokenizer=None):
    prompt=r"""
I will provide the description of three search actions, a search action history and a new search action, you should determine whether the new search action has been taken before. 

[Description of the search actions]

1. SearchWeb[Query]
    Description: Perform a query using the Google search engine to retrieve relevant information. You can use this tool to search for examples of API usage, API definitions, bug fixes, implementations of similar algorithms, and more.
    Parameters: Query (string): The search query to retrieve relevant information.
    Example: SearchWeb["How to implement a neural network in PyTorch"]

2. SearchFile[M]
    Description: Retrieve the content of a Python file from the current repository.
    Parameters: M (string): The name of the python file to search for in the current repository.
    Example: SearchFile["model.py"]

3. SearchCodeItem[M]  
    Description: Fetch information about a specific code item in the repository, including global variables, functions, methods, or classes. 
    Parameters: M (string): The name of the code item to search for in the current repository.
    Example: SearchCodeItem["Model"]
"""
    prompt += f"""
[Search Action History]

{action_history}

[Currrent Search Action]

{current_action}

Please check if the current search action has already been performed by referring to the search history. Follow these criteria:
1. If the current search action has been performed before, return True; otherwise, return False.
2. For SearchWeb action, consider the current action repeated if any history action serves the same function (even if not character-by-character identical) as the current search.
3. For SearchFile and SearchCodeItem actions, consider the current action repeated if the section title is character-by-character identical to a previous search.
4. Provide only True or False as your answer, with no additional comments or explanations.

[Your Answer]
""" 
    result_tmp = llm(prompt=prompt, model=model_name, generate_code=False, local_model=local_model, local_tokenizer=local_tokenizer)
    if 'R1' in model_name:
        result_tmp = result_tmp[1]
    return result_tmp

def extract_items(input_str: str):
    """
    Extract two items from the input string.
    
    Tries to find two substrings enclosed in double quotes.
    If that fails, it falls back to splitting on the comma.
    
    Example 1:
      Input: '"d(\\phi(x_{t+1}), z_k)", "The definition and properties of the function d ..."'
      Returns: ('d(\\phi(x_{t+1}), z_k)', 'The definition and properties of the function d ...')
      
    Example 2:
      Input: '$\\phi(x)$, "The definition of the class embedding function $\\phi(x)$"'
      Returns: ('$\\phi(x)$', 'The definition of the class embedding function $\\phi(x)$')
    """
    # First, attempt the original approach: two double-quoted items.
    items = re.findall(r'"(.*?)"', input_str)
    if len(items) >= 2:
        return items[0], items[1]
    
    # If not found, try to match a pattern where the first item is not quoted
    fallback = re.match(r'\s*([^,]+?),\s*"(.*?)"\s*$', input_str)
    if fallback:
        # The first group before the comma, and the second from the quotes.
        return fallback.group(1).strip(), fallback.group(2).strip()
    
    return None, None

def condtruct_prompt_for_retrieve_Concept(latex_code, latex_code_wait_retrieve, query):
    prompt=f"""
[Task Overview]
Extract information from the Target LaTeX Code that specifically addresses the query from the Source LaTeX Code.

[Source LaTeX Code]
{latex_code}

[Query]
{query}

[Target LaTeX Snippet]
{latex_code_wait_retrieve}

[Instructions]
1. Extract ONLY the portions of the Target LaTeX Snippet that directly address the query.
2. Maintain exact text and LaTeX formatting—do not paraphrase or add commentary.
3. For multiple information points, present each as a separate bullet point:
   * [Exact LaTeX snippet]
   * [Exact LaTeX snippet]
   For each information point, include sufficient surrounding context to ensure the meaning is clear and complete. Ensure each information point can be understood independently without losing its original meaning.
4. If nothing relevant is found, simply respond: "No relevant information found."

[Output Format]
Present only the extracted LaTeX content without additional explanation.
"""
    return prompt

def construct_prompt_PaperAgent(Section_String, latex_code):
    prompt = f"""
[Task Overview]
Reproduce Python code corresponding to a LaTeX-based methodology from a scientific paper. However, due to the paper’s length, it cannot be fully ingested by a large language model at once. Therefore, the solution requires two main steps:
1. Information Retrieval (Your Current Task): Extract relevant details, insights, and supporting information from the academic paper’s LaTeX description and related literature.
2. Code Reproduction (Subsequent Task): Implement the Python code based on the information gathered and the provided LaTeX.

[Your Specific Focus]
You are tasked exclusively with Step 1: Information Retrieval. You must gather and organize all necessary details that will later be used to implement the Python code. 

[Input]
1. List of sections: The paper includes the following sections (titles are provided for reference):

{Section_String}

2. LaTeX Description: The LaTeX code for the corresponding subsection in the paper, describing the algorithm implemented by the target function.

{latex_code}

3. Tools: Tools that can be adopted to gather external information during the information retrieval process.

    """
    
    prompt1 = r"""
* SearchPaper[query]
    Description: When a variable, concept, or any other element appears in the target section without its full definition or sufficient details, use this action to search for the complete information in the full paper.
    Parameters:
        - 'query' (string): A query describing the information that needs to be located within the full paper.
    Examples:
        - If the LaTeX contains: "We use the concept of $X_i$ to define Y," then the action should be: SearchPaper["The definition of $X_i$"]
        - If the LaTeX contains: "The function $f(x)$ is defined based on the properties of $\mathcal{G}$.", then the action should be: SearchPaper["The properties of $\mathcal{G}$"]

* SearchSection[x]
    Description: If the target section references another section in the paper with the title x, extract the information from the referenced section and return SearchSection[x].
    Parameters: 'x' (string): The title of the referenced section.
    Example: 
        - Latex: "The full derivation of our loss function can be found in method Section .", Action: SearchSection["method"]

* SearchLiterature[key, query]
    Description: If the target section cites another paper (\cite{label}) and you determine that some information needs to be retrieved from that paper, return SearchLiterature[label, query], where query is the specific information you need to look for in the referenced paper.
    Parameters: 
        - 'key' (string): The citation key of the referenced paper. In LaTeX, when citing a paper, we use \cite{x}, where x represents the citation key.
        - 'query' (string): The specific information to search for in the referenced paper.
    Example: I
        - Latex: "We adopt the metric proposed in \cite{wang2025}". Action: SearchLiterature["wang2025", "The proposed metric in the paper"]
        - Latex: "The algorithm is based on the work of \cite{smith2018}". Action: SearchLiterature["smith2018", "The algorithm details in the paper"]
        - Latex: "The dataset is based on the study by \cite{jones2020}". Action: SearchLiterature["jones2020", "The dataset details in the paper"]
        
[Instruction]
In order to complete code reproduction, it is first necessary to understand the algorithm described in the LaTeX description. The tools "SearchPaper", "SearchSection" and "SearchLiterature" should be used to retrieve relevant information from the paper to help you understand the methodology proposed in the latex description. For example:
    1. If the LaTeX Description lacks the definition of a variable, use "SearchPaper" tool to find its definition.
    2. If the LaTeX Description references other sections of the paper, use "SearchSection" tool to retrieve those sections and supplement the missing details.
    3. If the LaTeX Description cites methods from other papers, use "SearchLiterature" tool to extract relevant information from the referenced papers.
[Action]
    1. Apply a tool defined above to gather external information.
    2. If you have gathered all the necessary information, fully understood the LaTeX code, and are prepared to proceed to the Code Reproduction stage, the appropriate action is "Finish"

[Observation]
    1. If the action is apply predefined tool, then the observation should be the return response of the tool.

[Response Template]
    Thought: I think ...
    Action: SearchPaper[query] or SearchSection[label] or SearchLiterature[key, query] or Finish
    Observation: Outcome of the action.

[Your Answer]
    Please start information extraction step by step, strictly adhering to the provided template for the response format.
    """
    prompt += prompt1
    return prompt

def construct_prompt_CodeAgent(extracted_info, organization, Target_Function, latex_code, Python_File_Path, PaperInfomation_str, task_type='function'):
    if task_type == 'function':
        prompt = f"""
You are a code assistant tasked with reproducing a Python function corresponding to a algorithm in the methods part of a scientific paper. The local coding environment includes a GPU and supports CUDA. I will provide the following information:

1. Repository structure: The organization of files within the code repository. This is a repository-level code generation task, so you should explore the repo thoroughly to extract useful code.
2. Target function: The definition of the python function you need to implement.
3. LaTeX description: The LaTeX code for the corresponding algorithm in the paper, describing the algorithm implemented by the target function.
4. The extracted information: The information extracted from the target paper, and relevant literature that can provide you more details when implement the target function.
5. Tools: Tools that can be adopted to gather external information during the generation process.

[Repository Structure]

{organization}

[Target Function]

The target function is located at "{Python_File_Path}". Its definition consists of the following components:

1. Input Variables
2. Output Variables

The definition is as follows:

{Target_Function}

[LaTeX Description]

{latex_code}

[Extracted Information]

The information is extracted from the paper and relevant literature by a paper search agent, which consists of a series of information points. When you implement the target function, you should refer to the extracted information to understand the target algorithm. When information from "Relevant Literature" conflicts with the target paper, always prioritize the information from the target paper.

The extracted information is as follows:

{extracted_info}

[Tools]
    """
    
        prompt1 = r"""
1. SearchWeb[Query]
    Description: Perform a query using the Google search engine to retrieve relevant information. You can use this tool to search for examples of API usage, API definitions, bug fixes, implementations of similar algorithms, and more.
    Parameters: Query (string): The search query to retrieve relevant information.
    Example: SearchWeb["How to implement a neural network in PyTorch"]

2. SearchFile[M]
    Description: Retrieve the content of a Python file from the current repository.
    Parameters: M (string): The name of the python file to search for in the current repository.
    Example: SearchFile["model.py"]

3. SearchCodeItem[M]  
    Description: Fetch information about a specific code item in the repository, including global variables, functions, methods, or classes. 
    Parameters: M (string): The name of the code item to search for in the current repository.
    Example: SearchCodeItem["Model"]
        
Instruction:
In order to complete this task, it is necessary to use tools to search the code repository for context that can help implement the target function. For example:
    1. Use "SearchFile" to retrieve the content of a Python file from the repository.
    2. Use "SearchCodeItem" to find details about a specific code item within the repository.
    3. Use "SearchWeb" to retrieve information from the website.
    
To effectively tackle the code reproduction task, follow a structured process that alternates between Thought, Action, and Observation steps:

[Thought]
    1. Analyze the current situation.
    2. Identify missing information from code. As it is a repo-level code generation task, you need to explore the relvant functions, classes, in the code repository.
    3. Plan the next steps to gather the required information.

[Action]
    1. Apply a tool defined above to gather external information.
    2. If you are ready to generate the code, then the action should be "GenerateCode".

[Observation]
    1. If the action is apply predefined tool, then the observation should be the return response of the tool.
    2. If the action is "GenerateCode", then the observation is the result returned by the interpreter after executing the generated code.

[Response Template]
    Thought: I think ...
    Action: SearchWeb[Query] or SearchFile[M] or SearchCodeItem[M] or GenerateCode
    Observation: Outcome of the action.

[Implementation Guidelines]
    1. Step-by-step analysis of the LaTeX algorithm alongside extracted information.
    2. Comprehensive repository exploration using provided tools.
    3. Clean and efficient code implementation strictly matching the LaTeX algorithm.
    4. Adherence to the structured Thought, Action, Observation response format.

[Your Answer]
    """
        prompt += prompt1
    elif task_type == 'method':
        prompt = f"""
You are a code assistant tasked with reproducing a Python method within a class that corresponds to a algorithm in the methods part of a scientific paper. The local coding environment includes a GPU and supports CUDA. I will provide the following information:

1. Repository structure: The organization of files within the code repository. This is a repository-level code generation task, so you should explore the repo thoroughly to extract useful code.
2. Target method: The definition of the python method you need to implement.
3. LaTeX description: The LaTeX code for the corresponding algorithm in the paper, describing the algorithm implemented by the target method.
4. The extracted information: The information extracted from the target paper and relevant literature that can provide you more details when implement the target method.
5. Tools: Tools that can be adopted to gather external information during the generation process.

[Repository Structure]

{organization}

[Target Method]

The target method is located at "{Python_File_Path}". Its definition consists of the following components:

1. Input Variables
2. Output Variables

The definition is as follows:

{Target_Function}

[LaTeX Description]

{latex_code}

[Extracted Information]

The information is extracted from the paper and relevant literature by a paper search agent, which consists of a series of information points. When you implement the target function, you should refer to the extracted information to understand the target algorithm. When information from "Relevant Literature" conflicts with the target paper, always prioritize the information from the target paper.

The extracted information is as follows:

{extracted_info}

[Tools]
    """
        prompt1 = r"""
1. SearchWeb[Query]
    Description: Perform a query using the Google search engine to retrieve relevant information. You can use this tool to search for examples of API usage, API definitions, bug fixes, implementations of similar algorithms, and more.
    Parameters: Query (string): The search query to retrieve relevant information.
    Example: SearchWeb["How to implement a neural network in PyTorch"]

2. SearchFile[M]
    Description: Retrieve the content of a Python file from the current repository.
    Parameters: M (string): The name of the python file to search for in the current repository.
    Example: SearchFile["model.py"]

3. SearchCodeItem[M]  
    Description: Fetch information about a specific code item in the repository, including global variables, functions, methods, or classes. 
    Parameters: M (string): The name of the code item to search for in the current repository.
    Example: SearchCodeItem["Model"]
        
Instruction:
In order to complete this task, it is necessary to use tools to search the code repository for context that can help implement the target function. For example:
    1. Use "SearchFile" to retrieve the content of a Python file from the repository.
    2. Use "SearchCodeItem" to find details about a specific code item within the repository.
    3. Use "SearchWeb" to retrieve information from the website.
    
To effectively tackle the code reproduction task, follow a structured process that alternates between Thought, Action, and Observation steps:

[Thought]
    1. Analyze the current situation.
    2. Identify missing information from code. As it is a repo-level code generation task, you need to explore the relvant functions, classes, in the code repository.
    3. Plan the next steps to gather the required information.

[Action]
    1. Apply a tool defined above to gather external information.
    2. If you are ready to generate the code, then the action should be "GenerateCode".

[Observation]
    1. If the action is apply predefined tool, then the observation should be the return response of the tool.
    2. If the action is "GenerateCode", then the observation is the result returned by the interpreter after executing the generated code.

[Response Template]
    Thought: I think ...
    Action: SearchWeb[Query] or SearchFile[M] or SearchCodeItem[M] or GenerateCode
    Observation: Outcome of the action.

[Implementation Guidelines]
    1. Step-by-step analysis of the LaTeX algorithm alongside extracted information.
    2. Comprehensive repository exploration using provided tools.
    3. Clean and efficient code implementation strictly matching the LaTeX algorithm.
    4. Adherence to the structured Thought, Action, Observation response format.

[Your Answer]
    
    """     
        prompt += prompt1
    return prompt

def Extract_Function_Definition_withoutDescription(task, task_type, Python_File):
    if task_type == 'function' or task_type == 'method':
        function_definition = '\n'.join(Python_File[task['signature_position'][0]-1:task['signature_position'][1]])
        Function_string = function_definition + '\n'
    elif task_type == 'class':
        class_string = Python_File[task['signature_position'][0]-1:task['body_position'][1]]
        for i in range(len(task['subtasks'])-1, -1, -1):
            subtask = task['subtasks'][i]
            subtask_type = subtask['type']
            Function_string = Extract_Function_Definition_withoutDescription(subtask, subtask_type, Python_File)
            class_string += Function_string
        Function_string = class_string
    return Function_string

def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def FormatCheckAction_CodeAgent(action):
    action = action.strip()
    if action.startswith("GenerateCode"):
        return "GenerateCode"
    else:
        index_last = action.find("]")
        return action[:index_last+1]

def FormatCheckAction_PaperAgent(action):
    action = action.strip()
    if action.startswith("Finish"):
        return "Finish"
    else:
        index_last = action.find("]")
        return action[:index_last+1]

def remove_space(code_list):
    for m in range(len(code_list)-1, -1, -1):
        if code_list[m] == '':
            code_list.pop(m)
        else:
            code_list[m] = code_list[m].replace('\n', '')
    return code_list

def PaperAgent(sections, args, latex_project_directory, root_path, PaperRepoPath, prompt, latex_snippet):
    bib_output = read_bib_files(latex_project_directory, recursive=True)

    Already_Exist = list()
    Already_Exist.append(latex_snippet)

    n_calls, n_badcalls = 0, 0
    sections = Extract_Latex_Section(PaperRepoPath)
    for m in range(len(sections)-1, -1, -1):
        tmp = clean_latex_content(sections[m]['content'])
        if tmp == '':
            del sections[m]
    
    ThoughtList = list()
    search_history = list()
    Output = list()
    for i in range(30):
        ThoughtTimestep = dict()
        n_calls += 1
        prompt_input = prompt + "\nThought:"
        thought_action = llm(prompt_input, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None) 
        if 'R1' in args.model or 'r1' in args.model:
            think = thought_action[0]
            thought_action = thought_action[1]
        Loop = True
        while Loop:
            if n_badcalls == 10:
                prompt_intput = prompt + thought_action + AUGMENTLATEX
                code = llm(prompt_intput, model=args.model, effort=args.effort, generate_code=True, local_model=None, local_tokenizer=None)
                if 'R1' in args.model or 'r1' in args.model:
                    think = code[0]
                    code = code[1]
                for m in range(len(code)):
                    if code[m:m+9] == "```report":
                        code = code[m+9:]
                        break
                for m in range(len(code)-1, 0, -1):
                    if code[m-2:m+1] == "```":
                        code = code[:m-2]
                        break
                ThoughtTimestep['Observation'] = f"The generated latex is: {code} \n The number of bad calls reached 10."
                ThoughtTimestep['Thought'] = "Number of bad calls reached 10."
                ThoughtTimestep['Action'] = "Finish"
                if 'R1' in args.model or 'r1' in args.model:
                    ThoughtTimestep['inter_think'] = think
                return code, ThoughtList
            try:
                if args.model == 'deepseek-r1':
                    thought, action = thought_action.strip().split(f"Action:")
                else:
                    thought, action = thought_action.strip().split(f"\nAction:")
                action = FormatCheckAction_PaperAgent(action)
                if not action.startswith("SearchPaper") and not action.startswith("SearchSection") and not action.startswith("SearchLiterature") and not action.startswith("Finish"):
                    print('ohh... Error Parse...')
                    print(action)
                    prompt_intput = prompt + CORRECT_PaperAgent + "Thought:"
                    thought_action = llm(prompt_intput, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None)
                    if 'R1' in args.model or 'r1' in args.model:
                        think = thought_action[0]
                        thought_action = thought_action[1]
                    n_badcalls += 1
                    n_calls += 1
                    continue
                Loop = False
            except:
                print('ohh... Error Parse...')
                print(thought_action)
                n_badcalls += 1
                n_calls += 1
                prompt_intput = prompt + CORRECT_PaperAgent + "Thought:"
                thought_action = llm(prompt_intput, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None)
                if 'R1' in args.model or 'r1' in args.model:
                    think = thought_action[0]
                    thought_action = thought_action[1]

        repeat = False
        if action in search_history:
            repeat = True
        elif len(search_history) > 0 and not action.startswith("Finish") and action not in search_history:  
            search_string = ""
            num = 1
            for j in range(len(search_history)):
                search_string = search_string + str(num) + ". " + search_history[j] + "\n"
            judge = judge_repeat_action_PaperAgent(search_string, action, 'gpt-4o-mini', None, None)
            judge = judge.strip()
            correct = (judge == "True" or judge == "False")
            while not correct:
                judge = judge_repeat_action_PaperAgent(search_string, action, 'gpt-4o-mini', None, None)
                judge = judge.strip()
                correct = (judge == "True" or judge == "False")
            if judge == "True":
                repeat = True
        if not action.startswith("Finish"):
            search_history.append(action)
        try:
            result = ""
            if action.startswith("SearchPaper[") and action.endswith("]"):
                query = extract_label_paper(action) 
                query = query.strip()
                num = 1
                if repeat:
                    result = -1
                else:
                    result = "[Information Extracted From Target Paper Begin]\n"
                    for m in range(len(sections)):
                        input_retrieve = clean_latex_content(sections[m]['content'])
                        input_retrieve = remove_figures(input_retrieve)
                        prompt_retrieve = condtruct_prompt_for_retrieve_Concept(latex_snippet, input_retrieve, query)
                        result_tmp = llm(prompt_retrieve, model="gpt-4o-mini", generate_code=False, local_model=None, local_tokenizer=None)
                        if "No relevant information found." in result_tmp.strip():
                            continue
                        if result_tmp.endswith("<|eot_id|>"):
                            result_tmp = result_tmp[:-len("<|eot_id|>")]
                        result_tmp = result_tmp.strip()
                        if Is_Exist(Already_Exist, result_tmp):
                            continue
                        Already_Exist.append(result_tmp)
                        if result_tmp[-1] == "\n":
                            result_tmp = result_tmp[:-1]
                        result = result + result_tmp + "\n"
                        num += 1
                    result += "\n[Information Extracted From Target Paper End]\n"
                    output = f"""The relevant information of the query "{query}" wihin the paper is as follows:\n[Information Extracted From Target Paper Begin]\n{result}\n[Information Extracted From Target Paper End]\n"""
                    tmp = dict()
                    tmp['action'] = action
                    tmp['output'] = output
                    Output.append(tmp)
            elif action.startswith("SearchSection[") and action.endswith("]"):
                title  = extract_section(action)
                result = ""
                if repeat:
                    result = -1
                else:
                    for m in range(len(sections)):
                        if title == sections[m]['title']:
                            if sections[m]['retrieved'] == True:
                                result = -1
                                break
                            else:
                                sections[m]['retrieved'] =True
                                result = clean_latex_content(sections[m]['content'])
                                result = remove_figures(result)
                                Already_Exist.append(result)
                                break
                    
                    if result != "":
                        result = f"\n[Section Extracted From Target Paper Begin]\n{result}\n[Section Extracted From Target Paper End]\n" 
                        output = f"""The section with the title "{title}" wihin the paper is as follows:\n[Section Extracted From Target Paper Begin]\n{result}\n[Section Extracted From Target Paper End]\n"""
                    else:
                        output = f"""Cannot find the section with the title "{title}" wihin the paper."""
                        result = f"""Cannot find the section with the title "{title}" wihin the paper."""
                    tmp = dict()
                    tmp['action'] = action
                    tmp['output'] = output
                    Output.append(tmp)
            elif action.startswith("SearchLiterature[") and action.endswith("]"):
                name = action[len("SearchLiterature["):-1]
                bib_id, query = extract_items(name)
                bib_id = bib_id.strip()
                query = query.strip()
                if repeat:
                    result = -1
                else:
                    paper_title = bib_output[bib_id]
                    arxivID = Download_Latex_Repo(paper_title, root_path)
                    if arxivID == -1:
                        result = "The paper is not accessible via the ArXiv API, retrieval failed."           
                    else:
                        paper_Tmp_repo = os.path.join(root_path, 'ReleventLiterature')
                        paper_Tmp_repo = os.path.join(paper_Tmp_repo, arxivID)
                        sections_literature = Extract_Latex_Section(paper_Tmp_repo)
                        num = 1
                        for m in range(len(sections_literature)):
                            input_retrieve = clean_latex_content(sections_literature[m]['content'])
                            input_retrieve = remove_figures(input_retrieve)
                            prompt_retrieve = condtruct_prompt_for_retrieve_Concept(latex_snippet, input_retrieve, query)
                            result_tmp = llm(prompt_retrieve, model="gpt-4o-mini", generate_code=False, local_model=None, local_tokenizer=None)
                            if "No relevant information found." in result_tmp.strip():
                                continue
                            if Is_Exist(Already_Exist, result_tmp):
                                continue
                            Already_Exist.append(result_tmp)
                            if result_tmp[-1] == "\n":
                                result_tmp = result_tmp[:-1]
                            result = result + result_tmp + "\n"
                            num += 1
                        if result != "":
                            result = f"\n[Information Extracted From Relevant Literature Begin]\n{result}\n[Information Extracted From Relevant Literature End]\n"
                            output = f"""The pertinent details regarding "{query}" in the associated literature, identified by the citation key "{bib_id}", are as follows: [Information Extracted From Relevant Literature Begin]\n{result}\n[Information Extracted From Relevant Literature End]"""
                            tmp = dict()
                            tmp['action'] = action
                            tmp['output'] = output
                            Output.append(tmp)  
            elif action == "Finish": 
                index = thought_action.find("Finish")
                thought_action = thought_action[:index+1]
                prompt_intput = prompt + thought_action + AUGMENTLATEX
                code = llm(prompt_intput, model=args.model, effort=args.effort, generate_code=True, local_model=None, local_tokenizer=None)
                if 'R1' in args.model or 'r1' in args.model:
                    think = code[0]
                    code = code[1]
                code = code.strip()
                for m in range(len(code)):
                    if code[m:m+9] == "```report":
                        code = code[m+9:]
                        break
                for m in range(len(code)-1, 0, -1):
                    if code[m-2:m+1] == "```":
                        code = code[:m-2]
                        break
                ThoughtTimestep['Observation'] = f"The generated Latex is:\n{code}"
                ThoughtTimestep['Thought'] = thought
                ThoughtTimestep['Action'] = action
                if 'R1' in args.model or 'r1' in args.model:
                    ThoughtTimestep['inter_think'] = think
                return code, ThoughtList
            else:
                    print('ohh...', thought_action)
                    n_badcalls += 1
                    continue
        except Exception as e:
            print("Error:", e)
            n_badcalls += 1
            n_calls += 1
            prompt_intput = prompt + CORRECT_PaperAgent + "Thought:"
            thought_action = llm(prompt_intput, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None)
            if 'R1' in args.model or 'r1' in args.model:
                think = thought_action[0]
                thought_action = thought_action[1]

        if isinstance(result, str) and result.strip() == "":
            result = "No relevant information found."
            prompt = prompt + "Thought:\n\n" + thought_action + f"\nObservation:\n\n{result}\n"
        elif isinstance(result, int) and result == -1:
            if action.startswith("SearchPaper["):
                result = f"""The query "{query}" in the paper has already been retrieved. Please determine if there is any other missing information and proceed to the next round of operations."""
            elif action.startswith("SearchSection["):
                result = f"""The section with the title "{title}" in the paper has already been retrieved. Please determine if there is any other missing information and proceed to the next round of operations."""
            elif action.startswith("SearchLiterature["):   
                result = f"""The information related to the query "{query}" within the literature associated with the citation key "{bib_id}" has already been retrieved. Please determine if there is any other missing information and proceed to the next round of operations."""   
            prompt = prompt + "Thought:\n\n" + thought_action + f"\nObservation:\n\n{result}\n"
        else:
            prompt = prompt + "Thought:\n\n" + thought_action + f"\nObservation:\n\n{result}\n"
        ThoughtTimestep['Observation'] = result
        ThoughtTimestep['Thought'] = thought
        ThoughtTimestep['Action'] = action
        if 'R1' in args.model or 'r1' in args.model:
             ThoughtTimestep['inter_think'] = think
        ThoughtList.append(ThoughtTimestep)
        print("Thought:\n\n" + thought_action)
        print(f"\nObservation:\n\n{result}\n")

    prompt_intput = prompt + thought_action + AUGMENTLATEX
    code = llm(prompt_intput, model=args.model, effort=args.effort, generate_code=True, local_model=None, local_tokenizer=None)
    if 'R1' in args.model or 'r1' in args.model:
        think = code[0]
        code = code[1]
    for m in range(len(code)):
        if code[m:m+9] == "```report":
            code = code[m+9:]
            break
    for m in range(len(code)-1, 0, -1):
        if code[m-2:m+1] == "```":
            code = code[:m-2]
            break
    return code, ThoughtList

def check_gencode(thought_action):
    for m in range(len(thought_action)):
        if thought_action[m:m+9] == "```python":
            thought_action = thought_action[m+9:]
            return True
    
    return False

def CodeAgent(data, task_id, args, webTool, CodeRepoPath, CondaEnvName, gpu_id, Benchmark_path_global, prompt, file_name, latex_snippet, Function_string):    
    task_data = data['task_details'][task_id]
    all_files = []
    functionsAll = []
    classesAll = []   
    global_varsAll = []
    Already_Exist = list()
    Already_Exist.append(latex_snippet)
    importsAll = []
    name_list = list()
    for py_file in Path(CodeRepoPath).rglob("*.py"):
        if str(py_file).endswith('_backup.py'):
            continue    
        Tmp = dict()
        CodeRepoPath_tmp = CodeRepoPath.split('/')[:-1]
        CodeRepoPath_tmp = '/'.join(CodeRepoPath_tmp)
        functions, classes, global_vars, imports = analyze_file(py_file, False)
        Tmp['functions'] = functions
        Tmp['classes'] = classes
        Tmp['global_vars'] = global_vars
        Tmp['imports'] = imports
        Tmp['Path'] = str(py_file)[len(CodeRepoPath_tmp)+1:]
        all_files.append(Tmp)
        functionsAll.extend(functions)
        classesAll.extend(classes)
        global_varsAll.extend(global_vars)
        importsAll.extend(imports)
        for k in range(len(functions)):
            name_list.append(functions[k]['name'])
        for k in range(len(classes)):
            name_list.append(classes[k]['name'])
            for l in range(len(classes[k]['methods'])):
                name_list.append(classes[k]['methods'][l]['name'])
        for k in range(len(global_varsAll)):
            name_list.append(global_varsAll[k]['name'])

    n_calls, n_badcalls = 0, 0
    ThoughtList = list()
    CodeGenTimes = 0
    search_history = list()
    Output = list()
    CodeList = list()
    overrun_time = 0
    for i in range(50):
        ThoughtTimestep = dict()
        n_calls += 1
        prompt_input = prompt + "\nThought:"
        thought_action = llm(prompt_input, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None)
        if 'R1' in args.model or 'r1' in args.model:
            think = thought_action[0]
            thought_action = thought_action[1]
        Loop = True
        while Loop:
            if n_badcalls == 10:
                prompt1 = prompt + thought_action + GENCODE
                code = llm(prompt1, model=args.model,effort=args.effort, generate_code=True, local_model=None, local_tokenizer=None)
                if 'R1' in args.model or 'r1' in args.model:
                    think = code[0]
                    code = code[1]
                code = code.strip()
                for m in range(len(code)):
                    if code[m:m+9] == "```python":
                        code = code[m+9:]
                        break
                for m in range(len(code)-1, 0, -1):
                    if code[m-2:m+1] == "```":
                        code = code[:m-2]
                        break
                ThoughtTimestep['Observation'] = f"The generated code is: {code} \n The number of bad calls reached 10."
                ThoughtTimestep['Thought'] = "Number of bad calls reached 10."
                ThoughtTimestep['Action'] = "GenerateCode"
                if 'R1' in args.model or 'r1' in args.model:
                    ThoughtTimestep['inter_think'] = think
                ThoughtList.append(ThoughtTimestep)
                return code, ThoughtList
            try:
                if check_gencode(thought_action):
                    thought = None
                    action = "GenerateCode"
                else:
                    thought, action = thought_action.strip().split(f"Action:")
                action = FormatCheckAction_CodeAgent(action)
                if not action.startswith("SearchFile") and not action.startswith("SearchCodeItem") and not action.startswith("SearchWeb") and not action.startswith("GenerateCode"):
                    print('ohh... Error Parse...')
                    print(action)
                    prompt_intput = prompt + CORRECT + "Thought:"
                    thought_action = llm(prompt_intput, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None)
                    if 'R1' in args.model or 'r1' in args.model:
                        think = thought_action[0]
                        thought_action = thought_action[1]
                    n_badcalls += 1
                    n_calls += 1
                    continue
                Loop = False
            except:
                print('ohh... Error Parse...')
                print(thought_action)
                n_badcalls += 1
                n_calls += 1
                prompt_intput = prompt + CORRECT + "Thought:"
                thought_action = llm(prompt_intput, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None)
                if 'R1' in args.model or 'r1' in args.model:
                    think = thought_action[0]
                    thought_action = thought_action[1]

        repeat = False
        if action in search_history:
            repeat = True
        elif len(search_history) > 0 and not action.startswith("GenerateCode") and action not in search_history:  
            search_string = ""
            num = 1
            for j in range(len(search_history)):
                search_string = search_string + str(num) + ". " + search_history[j] + "\n"
                num += 1
            judge = judge_repeat_action_CodeAgent(search_string, action, 'gpt-4o-mini', None, None)
            judge = judge.strip()
            correct = (judge == "True" or judge == "False")
            while not correct:
                judge = judge_repeat_action_CodeAgent(search_string, action, 'gpt-4o-mini', None, None)
                judge = judge.strip()
                correct = (judge == "True" or judge == "False")
            if judge == "True":
                repeat = True
        if not action.startswith("GenerateCode"):
            search_history.append(action)
        try:
            result = ""
            if action.startswith("SearchFile[") and action.endswith("]"):
                name = extract_pythonfile(action)
                if repeat:
                    result = -1
                else:
                    if not name.endswith('.py') and not name.endswith('.json') and not name.endswith('.txt') and not name.endswith('.md'):
                        result = "It is not a valid file name."  
                    else:   
                        files = list_all_files(CodeRepoPath)
                        target_file_path = None
                        for file in files:
                            if file.endswith(name):
                                target_file_path = file
                                break
                        if target_file_path is None:
                            result = f"""The file "{name}" does not exist in the repository."""
                        else:
                            with open(target_file_path, 'r') as f:
                                file_content = f.read()
                            result = f"""The content of the python file f"{name}" is as follows:\n""" + file_content + '\n' 
                            
                            for num in range(len(functions)):
                                if functions[num]['path'].endswith(name):
                                    functions[num]['visited'] = True
                            for num in range(len(classes)):
                                if classes[num]['path'].endswith(name):
                                    classes[num]['visited'] = True
                                    for num_method in range(len(classes[num]['methods'])):
                                        classes[num]['methods'][num_method]['visited'] = True          
            elif action.startswith("SearchCodeItem[") and action.endswith("]"):
                name = extract_code_item(action)
                if repeat:
                    result = -1
                else:
                    function_definition_target = Extract_Function_Definition_withoutDescription(task_data, task_data['type'], task_data['ori_python_file'])
                    function_definition_target = function_definition_target.strip()
                    index1 = function_definition_target.index('def')
                    index2 = function_definition_target.index('(')
                    function_definition_target = function_definition_target[index1+3:index2].strip()
                    if name in function_definition_target:
                        result = "It is yet to be implemented, and detailed information cannot be provided at this time."
                    else:
                        name = name.strip()
                        name = name.split('.')[-1]
                        CodeRepoPath_tmp = CodeRepoPath.split('/')[-1]
                        
                        result_function = ""
                        for m in range(len(functionsAll)):
                            if functionsAll[m]['name'] == name and functionsAll[m]['visited'] == True:
                                result_function = f"""The function "{name}" has already been included in the prompt.\n"""
                            if functionsAll[m]['name'] == name and functionsAll[m]['visited'] == False:
                                functionsAll[m]['visited'] = True
                                result_function = f"""The function "{name}" is defined in the file located at "{functionsAll[m]['path']}". Its definition is as follows:\n\n{functionsAll[m]['definition']}"""
                                break
                        
                        result_function_class = ""
                        result_method = ""
                        for m in range(len(classesAll)):
                            if  classesAll[m]['name'] == name and classesAll[m]['visited'] == True:
                                result_function_class = f"""The class "{name}" has already been included in the prompt.\n"""
                            if classesAll[m]['name'] == name and classesAll[m]['visited'] == False:
                                result_function_class = classesAll[m]['definition']
                                method_name_list = list()
                                for n in range(len(classesAll[m]['methods'])):
                                    method_name_list.append(classesAll[m]['methods'][n]['name'])
                                if function_definition_target in method_name_list:
                                    for n in range(len(classesAll[m]['methods'])):
                                        if classesAll[m]['methods'][n]['name'] == function_definition_target:
                                            function_definition_target = classesAll[m]['methods'][n]['definition']
                                    result_function_class = result_function_class.replace(function_definition_target, Function_string)
                                result_function_class = f"""
    The class "{name}" is defined in the file "{classesAll[m]['path']}". 

    Its definition is as follows:

    {result_function_class}
    """
                                classesAll[m]['visited'] = True
                                break

                            for n in range(len(classesAll[m]['methods'])):
                                if classesAll[m]['methods'][n]['name'] == name and classesAll[m]['methods'][n]['visited'] == True:
                                    result_method = f"""The method "{name}" defined in class "{classesAll[m]['name']}" has been retrieved before.\n"""
                                if classesAll[m]['methods'][n]['name'] == name and classesAll[m]['methods'][n]['visited'] == False:
                                    result_method = f"""The method "{name}" defined in class "{classesAll[m]['name']}" is defined in the file located at "{classesAll[m]['methods'][n]['path']}". Its definition is as follows:\n\n{classesAll[m]['methods'][n]['definition']}\n"""
                                    classesAll[m]['methods'][n]['visited'] = True
                                    break
                        
                        
                        result_global_variable = ""
                        for m in range(len(global_varsAll)):
                            if global_varsAll[m]['name'] == name and global_varsAll[m]['visited'] == True:
                                result_global_variable = f"""The global variable "{name}" has already been included in the prompt.\n"""
                            if global_varsAll[m]['name'] == name and global_varsAll[m]['visited'] == False:
                                global_varsAll[m]['visited'] = True
                                result_global_variable = f"""The global variable "{name}" is defined in the file located at "{global_varsAll[m]['path']}". Its definition is as follows:\n\n{global_varsAll[m]['definition']}\n"""
                                break
                        if result_function == "" and result_function_class == "" and result_method == "" and result_global_variable == "":
                            result = f"""No relevant information found for the code item "{name}". It may be a variable. Please try searching for a class, function, global variable, or method instead."""
                        else:
                            result = result_function + result_function_class + result_method + result_global_variable 
            elif action.startswith("SearchWeb[") and action.endswith("]"):
                input_query  = extract_label_web(action)
                if repeat:
                    result = -1
                else:
                    result = webTool.WebsearchGoogle(input_query, site_filter=data['repo_original_url'])
                    output = f"""The relevant information of the query "{input_query}" searched through internet is as follows:\n{result}"""
                    tmp = dict()
                    tmp['action'] = action
                    tmp['output'] = output
                    Output.append(tmp)
            elif action == "GenerateCode":
                index = thought_action.find("GenerateCode")
                thought_action = thought_action[:index+1]
                prompt1 = prompt + thought_action + GENCODE
                code = llm(prompt1, model=args.model,effort=args.effort, generate_code=True, local_model=None, local_tokenizer=None)
                if 'R1' in args.model or 'r1' in args.model:
                    think = code[0]
                    code = code[1]
                print(code)
                code = code.strip()
                for m in range(len(code)):
                    if code[m:m+9] == "```python":
                        code = code[m+9:]
                        break
                for m in range(len(code)-1, 0, -1):
                    if code[m-2:m+1] == "```":
                        code = code[:m-2]
                        break
                
                if code in CodeList:
                    ThoughtTimestep['Observation'] = f"The generated code is: {code} \n The code has been generated before."
                    ThoughtTimestep['Thought'] = thought
                    ThoughtTimestep['Action'] = action
                    if 'R1' in args.model or 'r1' in args.model:
                        ThoughtTimestep['inter_think'] = think
                    ThoughtList.append(ThoughtTimestep)
                    return code, ThoughtList

                CodeList.append(code)
                Success, feedback = run_code(code, data, task_id, CodeRepoPath, CondaEnvName, gpu_id, file_name, task_data['script'], Benchmark_path_global, args.conda_env_path)

                CodeGenTimes += 1
                if Success:
                    ThoughtTimestep['Observation'] = f"The generated code is: {code} \n The feedback from the interpreter is:\n Success"
                    ThoughtTimestep['Thought'] = thought
                    ThoughtTimestep['Action'] = action
                    if 'R1' in args.model or 'r1' in args.model:
                        ThoughtTimestep['inter_think'] = think
                    ThoughtList.append(ThoughtTimestep)
                    return code, ThoughtList
                else:   
                    if feedback == "Execution timed out after 15 minutes. Far too long!":
                        overrun_time += 1
                    if overrun_time == 3:
                        ThoughtTimestep['Observation'] = f"The generated code is: {code} \n The code generation has overruned 3 times and no feedback from the interpreter."
                        ThoughtTimestep['Thought'] = thought
                        ThoughtTimestep['Action'] = action
                        ThoughtList.append(ThoughtTimestep)
                        if 'R1' in args.model or 'r1' in args.model:
                            ThoughtTimestep['inter_think'] = think
                        return code, ThoughtList
                    if CodeGenTimes > 10:
                        ThoughtTimestep['Observation'] = f"The generated code is: {code} \n The code generation has failed 10 times and no feedback from the interpreter."
                        ThoughtTimestep['Thought'] = thought
                        ThoughtTimestep['Action'] = action
                        if 'R1' in args.model or 'r1' in args.model:
                            ThoughtTimestep['inter_think'] = think
                        ThoughtList.append(ThoughtTimestep)
                        return code, ThoughtList
                    result = f"The generated code is: {code} \n The feedback from the interpreter is:\n {feedback}"
            else:
                    print('ohh...', thought_action)
                    n_badcalls += 1
                    continue
        except Exception as e:
            print("Error:", e)
            n_badcalls += 1
            n_calls += 1
            prompt_intput = prompt + CORRECT + "Thought:"
            thought_action = llm(prompt_intput, model=args.model, effort=args.effort, stop=[f"\nObservation"], local_model=None, local_tokenizer=None)
            if 'R1' in args.model or 'r1' in args.model:
                think = thought_action[0]
                thought_action = thought_action[1]

        if isinstance(result, str) and result.strip() == "":
            result = "No relevant information found."
            prompt = prompt + "Thought:\n\n" + thought_action + f"\nObservation:\n\n{result}\n"
        elif isinstance(result, int) and result == -1:
            if action.startswith("SearchFile["):
                result = f"""The file "{name}" within the repo has already been retrieved. Please determine if there is any other missing information and proceed to the next round of operations."""
            elif action.startswith("SearchCodeItem["):
                result = f"""The code item "{name}" has already been retrieved. Please determine if there is any other missing information and proceed to the next round of operations."""
            elif action.startswith("SearchWeb["):
                result = f"""The response to the query "{input_query}" has been retrieved from the internet. Please determine if there is any other missing information and proceed to the next round of operations."""
                 
            prompt = prompt + "Thought:\n\n" + thought_action + f"\nObservation:\n\n{result}\n"
        else:
            prompt = prompt + "Thought:\n\n" + thought_action + f"\nObservation:\n\n{result}\n"
        ThoughtTimestep['Observation'] = result
        ThoughtTimestep['Thought'] = thought
        ThoughtTimestep['Action'] = action
        if 'R1' in args.model or 'r1' in args.model:
             ThoughtTimestep['inter_think'] = think
        ThoughtList.append(ThoughtTimestep)
        print("Thought:\n\n" + thought_action)
        print(f"\nObservation:\n\n{result}\n")
    
    prompt = prompt + GENCODE
    code = llm(prompt, model=args.model, effort=args.effort, generate_code=True, local_model=None, local_tokenizer=None)
    if 'R1' in args.model or 'r1' in args.model:
        think = code[0]
        code = code[1]
    code = code.strip()
    for m in range(len(code)):
        if code[m:m+9] == "```python":
            code = code[m+9:]
            break
    for m in range(len(code)-1, 0, -1):
        if code[m-2:m+1] == "```":
            code = code[:m-2]
            break
    ThoughtTimestep['Observation'] = f"The generated code is: {code} \n The code generation has failed 10 times and no feedback from the interpreter."
    ThoughtTimestep['Thought'] = thought
    ThoughtTimestep['Action'] = action
    if 'R1' in args.model or 'r1' in args.model:
        ThoughtTimestep['inter_think'] = think
        ThoughtList.append(ThoughtTimestep)
    return code, ThoughtList

def main(args):
    Benchmark_path_global = args.root_path
    benchpath = os.path.join(Benchmark_path_global, "Benchmark")
    benchmark_dirs = [d for d in os.listdir(benchpath) 
                     if os.path.isdir(os.path.join(benchpath, d))]
    DataPath = os.path.join(Benchmark_path_global, "Data.json")
    with open(DataPath, 'r') as f:
        Data = json.load(f)
    gpu_id = args.gpu_id
    webTool = WebSearch(GPT_model="gpt-4o-mini")
    
    for k in range(0, 36):
        benchmark_path = ""
        for tmp in benchmark_dirs:
            if tmp.startswith(str(k)+ '-'):
                benchmark_path = tmp
                benchmark_path = os.path.join(Benchmark_path_global, benchmark_path)
                break
        if args.model == 'o3-mini':
            OutputDir = os.path.join(args.OutputDir, str(k), "SciReproducer_" + args.model+ "_" + args.effort) 
        else:
            OutputDir = os.path.join(args.OutputDir, str(k), "SciReproducer_" + args.model) 

        if os.path.exists(OutputDir):
            shutil.rmtree(OutputDir)
            os.makedirs(OutputDir)
        else:
            os.makedirs(OutputDir)

        data = Data[k]
        Benchmark_path = benchmark_path
        repo_path = data['project_path']
        file_organization = data['file_organization']
        answer = list()
        CondaEnvName = data['enviorment_name']
        CodeRepoPath = os.path.join(Benchmark_path_global, repo_path)
        PaperRepoPath = os.path.join(Benchmark_path_global, data['latex_code_path'])

        for i in range(0, len(data['task_details'])):
            task = data['task_details'][i]
            task_type = task['type']
            recover(data, i, Benchmark_path_global)
            file_path = os.path.join(CodeRepoPath, task['completion_path'][2:])
            latex_code = task['latex_code']
            Function_string = Extract_Function_Definition(task, task['ori_python_file'])
            Python_File_Path = repo_path[1:].split('/')[-1] + task['completion_path'][1:]

            # Paper Agent
            sections = Extract_Latex_Section(PaperRepoPath)
            for m in range(len(sections)-1, -1, -1):
                tmp = clean_latex_content(sections[m]['content'])
                if tmp == '':
                    del sections[m]
            Section_String = ""
            for m in range(len(sections)):
                Section_String = Section_String + "* " + sections[m]['title'] + "\n"
            prompt = construct_prompt_PaperAgent(Section_String, latex_code)
            print(prompt)
            file_path = os.path.join(CodeRepoPath, task['completion_path'][2:])
            response, ThoughtList = PaperAgent(sections, args, PaperRepoPath, Benchmark_path_global, PaperRepoPath, prompt, latex_code)
            print(response)
            Output = dict()

            # Store the results for paper agent
            tmp = dict()
            tmp['answer'] = response
            tmp['ThoughtList'] = ThoughtList
            Output['PaperAgent'] = tmp
            prompt = construct_prompt_CodeAgent(response, file_organization, Function_string, latex_code, Python_File_Path, task_type)
            print(prompt)
            response, ThoughtList = CodeAgent(data, i, args, webTool, CodeRepoPath, CondaEnvName, gpu_id, Benchmark_path_global, prompt, file_path, latex_code, Function_string)
            
            # Store the results for code agent
            tmp = dict()
            tmp['prompt'] = prompt
            tmp['answer'] = response
            tmp['ThoughtList'] = ThoughtList
            Output['CodeAgent'] = tmp

            # Store the final answer
            Output['answer'] = response
            answer.append(Output)
            output_file = os.path.join(OutputDir, f"task{str(i+1)}" + ".pkl")
            output_file = os.path.join(Benchmark_path, output_file)
            with open(output_file, 'wb') as f:
                pickle.dump(Output, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='gpt-4o-mini', choices=['gpt-4o', 'gpt-4o-mini', 'o3-mini', 'deepseek-r1', 'deepseek-v3', 'claude-3-7', 'gemini-2.0-flash', 'gemini-2.0-flash-thinking'])
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--effort', default="high", type=str)
    parser.add_argument('--root_path', default='/scratch/prj/intelmo/Project/SciCodeGen/Sci-Reproducer/', type=str)
    parser.add_argument('--Benchmark_Path', default="/scratch/prj/intelmo/Project/SciCodeGen/Sci-Reproducer/Benchmark/", type=str)
    parser.add_argument('--OutputDir', default="/scratch/prj/intelmo/Project/SciCodeGen/Sci-Reproducer/Result", type=str)
    parser.add_argument('--conda_env_path', default="/scratch/prj/intelmo/Project/SciCodeGen/Sci-Reproducer/envs_sci", type=str)
    args = parser.parse_args()
    result = main(args)
    

