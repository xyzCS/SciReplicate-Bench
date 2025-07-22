# SciReplicate-Bench (COLM 2025)

Code release for [SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers](https://arxiv.org/abs/2504.00255). [Conference on Language Modeling 2025]

## File Organization
```
SciReplicate-Bench/
├── utils/                                           # Core utilities for SciReproducer and evaluation metrics
│   ├── __init__.py                                  # Package initialization file
│   ├── CodeAgentTools.py                            # Tools and utilities for the code agent
│   ├── PaperAgentTools.py                           # Tools and utilities for the paper agent
│   ├── WebSearch.py                                 # Web search functionality
│   ├── utils.py                                     # General utility functions
│   └── Reason_Process_ACC.py                        # Reasoning graph accuracy calculation
│
├── scripts/                                         # Setup and utility scripts
│   └── env.sh                                       # Script to extract and setup conda environments
│
├── envs_sci/                                        # Extracted conda environments (created by env.sh)
│   ├── ColdFusion/                                  # Python environment for ColdFusion paper
│   ├── order/                                       # Python environment for order paper
│   ├── gac_env/                                     # Python environment for GAC paper
│   └── ...                                          # Additional environments (36 total)
│
├── Benchmark/                                       # Code repositories for all benchmark papers
│   ├── 0-coldfusion/                                # Source code repository for ColdFusion paper
│   ├── 1-order/                                     # Source code repository for order paper
│   ├── 2-gac/                                       # Source code repository for GAC paper
│   └── ...                                          # Additional repositories (36 total)
│
├── Result/                                          # Experiment results and outputs after running the SciReproducer 
│   ├── 0/                                           # Results for paper 0 (ColdFusion)
│   │   └── SciReproducer_gpt-4o-mini/               # Generated code using GPT-4o-mini
│   │       ├──task1.pickle                          # Code output for ColdFusion task 1
│   │       └── ...
│   └── ...                                          # Results for all papers
│
├── Data.json                                        # Main dataset containing paper information and tasks
├── Evaluation.py                                    # Evaluation metrics (CodeBLEU, Execution Accuracy, etc.)
├── SciReproducer.py                                 # Main dual-agent framework implementation
├── envs_sci.zip                                     # Archive containing all conda environments
└── README.md                                        # Project documentation
```

## 1. Setting Up Python Environments for All Papers

### Prerequisites
- Conda/Miniconda installed on your system
- `envs_sci.zip` file available in the root directory ([link](https://emckclac-my.sharepoint.com/:u:/g/personal/k23069329_kcl_ac_uk/EeMT-H7rvBtGv11oZIKHbv4BKKmAozX_bRAWwX59ND7ahw?e=I7ARfS))
- `Benchmark` directory. ([link](https://emckclac-my.sharepoint.com/:u:/g/personal/k23069329_kcl_ac_uk/ESsNHkIvAdVOhWDWSc0tWkQBd1RS6mRot--3jaN_VgDM3Q?e=AYm2rH), please download it and unzip it)
- Hardware Requirements:
    - Sufficient disk space for extracting 36 conda environments.
    - Ubuntu operating system.
    - CUDA Version: 12.2
    - GPU: A single NVIDIA A100 (80GB) GPU is required to execute all code repositories associated with the benchmark papers.

### Setup Instructions
```bash
cd root_path
bash ./scripts/env.sh root_path
```
After setup all the environments, run the reference code (refer to section 4.2.1) to make sure all code repositories can run correctly.

#### Parameters
- `--root_path`: Path to the root directory containing all project files. (Path to the 'SciReplicate-Bench' dir)

## 2. Setting Up Python Environments for SciReproducer
```
conda env create -f environment.yml

or

conda env create -f environment.yml -p path_target_env
```

### Parameters
- `path_target_env`: Path to the conda environments. (For example, {path_to_anaconda3}/envs/codegen)

## 3. Run the SciReproducer

### Prerequisites
- All conda environments set up (from step 1)
- SciReproducer environment activated (from step 2)
- API key configured for the chosen model
  - For web search tools, please follow the instructions in Section 5 to apply for a Google Search API key and a CSE ID.

### Usage

#### Step 1: Set API Keys
```bash

# Hugginface login, and you need to apply the authentication for accessing models.
huggingface-cli login

# For Web Search Tools. (Refer to section 5 for guidance)
export GoogleSearch_API_KEY="GoogleSearch_API_KEY"
export GoogleSearch_CSEID="your_GoogleSearch_CSEID_here"

# For LLMs
export OPENAI_API_KEY="your_openai_key_here"        # Required for OpenAI models
export DEEPSEEK_API_KEY="your_deepseek_key_here"    # Optional, for DeepSeek models
export CLAUDE_API_KEY="your_claude_key_here"        # Optional, for Claude models
export GEMINI_API_KEY="your_gemini_key_here"        # Optional, for Gemini models
```

#### Step 2: Run SciReproducer
```bash
bash ./scripts/run_sci_reproducer.sh <root_path> [model]
```

#### Parameters
- `<root_path>`: Path to the root directory containing all project files (the 'SciReplicate-Bench' directory)
- `[model]`: (Optional) The language model to use. Default: `gpt-4o-mini`

#### Supported Models
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o3-mini`
- **DeepSeek**: `deepseek-r1`, `deepseek-v3`
- **Anthropic**: `claude-3-5-sonnet`
- **Google**: `gemini-2.0-flash`, `gemini-2.0-flash-thinking`

#### Example
```bash
# Using default model (gpt-4o-mini)
bash ./scripts/run_sci_reproducer.sh /path/to/SciReplicate-Bench

# Using specific model
bash ./scripts/run_sci_reproducer.sh /path/to/SciReplicate-Bench gpt-4o
```

### Output Structure
The results will be saved in the specified output directory with the following structure:
```
Result/
├── 0/                                  # Results for paper 0 (ColdFusion)
│   └── SciReproducer_{model}/          # Generated code directory
│       ├──task1.pickle                 # Generated code for task 1 within ColdFusion
│       └── ...
└── ...                                 # Results for all papers
```

### Toy Example
We provide a toy example in the toy.py file, which includes the following components:
- Code Generation Prompt Template (GENCODE): Defines a detailed prompt template to guide LLMs in generating code that adheres to a specific format for calculating reasoning graph accuracy.
- Main Function (main): Iterates through 36 benchmark code repositories (repo_id 0–35). For each repository and each task, it outlines the step-by-step process for handling the task.

## 4. Evaluation Metrics

After running SciReproducer, you can evaluate the generated code using 4 different metrics.

### Basic Command Structure
```bash
python Evaluation.py --metric <metric_name> --model <model_name> --root_path <root_path> --result_path <result_path> [additional_options]
```

#### Common Parameters
- `--metric`: Type of evaluation metric to calculate, chosing from ['CodeBLEU_Score', 'execution_ACC', 'Recall', 'ReasoningGraph_ACC'].
- `--model`: Model name used for code generation.
- `--root_path`: Path to the root directory containing all project files (the 'SciReplicate-Bench' directory).
- `--gpu_id`: GPU ID to use for execution (default: 0)

---

### 4.1 CodeBLEU Score|Recall|Reasoning Graph Acc

```bash
export OPENAI_API_KEY="your_openai_key_here"        
python Evaluation.py \
  --metric [CodeBLEU_Score|Recall|ReasoningGraph_ACC] \
  --model gpt-4o-mini \
  --root_path /path/to/SciReplicate-Bench \
  --result_path /path/to/SciReplicate-Bench/Result
```

---

### 4.2 Execution Accuracy

#### 4.2.1 Obtain the output of the reference code. Due to the difference of different machines, you need to run the reference code on your machine to obtain the reference output.
 
```bash
export OPENAI_API_KEY="your_openai_key_here"
python Evaluation.py \
  --metric execution_ACC \
  --model gpt-4o-mini \
  --gpu_id 0 \
  --root_path /path/to/SciReplicate-Bench \
  --reference \
```


#### 4.2.2 Evaluate the output of the generated code. Obtain the output of the generated code and compare the generated output with the reference output.

```bash
export OPENAI_API_KEY="your_openai_key_here"
python Evaluation.py \
  --metric execution_ACC \
  --model gpt-4o-mini \
  --gpu_id 0 \
  --root_path /path/to/SciReplicate-Bench \
```

---

## 5. Google Search API Setup

### How to Get These Credentials:

**Google Search API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Custom Search JSON API"
4. Go to "Credentials" → "Create Credentials" → "API Key"

**Google Custom Search Engine ID (CSE ID):**
1. Go to [Google Custom Search](https://cse.google.com/)
2. Create a new search engine
3. Set it to search the entire web
4. Copy the Search Engine ID from the control panel

### Setting the Environment Variables:
```bash
export GoogleSearch_API_KEY="your_google_search_api_key"
export GoogleSearch_CSEID="your_custom_search_engine_id"
```


## Reference
```
@article{xiang2025scireplicate,
  title={Scireplicate-bench: Benchmarking llms in agent-driven algorithmic reproduction from research papers},
  author={Xiang, Yanzheng and Yan, Hanqi and Ouyang, Shuyin and Gui, Lin and He, Yulan},
  journal={arXiv preprint arXiv:2504.00255},
  year={2025}
}
```
