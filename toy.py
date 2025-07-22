import os
from utils.CodeAgentTools import run_code
from utils.utils import recover
import pickle
import json
import shutil

import argparse

# ⚠️ Important: This is the prompt for generating the code that follows the pre-defined format.
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


def main(args):
    root_path = args.root_path
    benchpath = os.path.join(root_path, "Benchmark")
    benchmark_dirs = [d for d in os.listdir(benchpath) 
                     if os.path.isdir(os.path.join(benchpath, d))]
    DataPath = os.path.join(root_path, "Data.json")
    with open(DataPath, 'r') as f:
        Data = json.load(f)
    
    # Iterate through each code repository (36 in total)
    for repo_id in range(0, 36):
        benchmark_path = ""
        for tmp in benchmark_dirs:
            if tmp.startswith(str(repo_id)+ '-'):
                benchmark_path = tmp
                benchmark_path = os.path.join(root_path, benchmark_path)
                break
        
        # Create output directory for the current code repository
        OutputDir = os.path.join(args.OutputDir, str(repo_id), "SciReproducer_" + args.model) 
        
        # If the output directory exists, remove it and create a new one
        if os.path.exists(OutputDir):
            shutil.rmtree(OutputDir)
            os.makedirs(OutputDir)
        else:
            os.makedirs(OutputDir)
    
        # Load the data for the current code repository
        data = Data[repo_id]
        Benchmark_path = benchmark_path
        repo_path = data['project_path']
        CondaEnvName = data['enviorment_name']
        CodeRepoPath = os.path.join(root_path, repo_path)

        # Iterate through each task for each code repository
        for task_id in range(0, len(data['task_details'])):
            # Step 1 ⚠️ Important: Before starting a task, recover the data for the current task and remove the reference code from the repository to prevent the LLM from accessing it when searching for relevant information within the codebase.
            recover(data, task_id, root_path)
            file_path = os.path.join(CodeRepoPath, data['completion_path'][2:])
            # Step 2: Then you apply your approach to generate the code. 
            # Assume the generated code is stored in the variable `generated_code`
            generated_code = f"""
def assign_observations_to_classes(cur_trainset, in_scope_topic_features):
    index = faiss.IndexFlatL2(cur_trainset.shape[1])
    index.add(in_scope_topic_features)

    # ---------------------------------------------------------------------------
    # Snippet 1: For every observation in cur_trainset, search for the single
    # nearest class vector (k=1) based on their Euclidean distances. The returned I array gives the index of the
    # closest class feature (i.e., \(\arg\min_k d(\phi(x), \phi(c_k))\)).
    # ---------------------------------------------------------------------------
    # [Begin Snippet 1]
    _, I = index.search(cur_trainset, 1)
    I = I[:, 0]
    # [End Snippet 1]

    return I
"""
            
            # Step 3: You can run the code using the `run_code` function.
            Success, feedback = run_code(generated_code, data, task_id, CodeRepoPath, CondaEnvName, args.gpu_id, file_path, data['script'], root_path, args.conda_env_path)


            # Step 4: Save the generated code to the output directory.
            Output = dict()
            Output['answer'] = generated_code
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