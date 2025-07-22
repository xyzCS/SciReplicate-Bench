import re
import arxiv
import os
import tarfile
import wget
import requests
import urllib.parse
import pymupdf

def extract_section(action: str) -> str:
    """
    Extract the label from a SearchSection action.
    
    This function supports actions in the following formats:
        - SearchSection[a]
        - SearchSection["a"]
    """
    pattern = r'^SearchSection\[\s*"?([^"\]]+)"?\s*\]$'
    match = re.match(pattern, action)
    if match:
        return match.group(1).strip()
    return None

def get_arxiv_id_by_title(title):
    # Initialize the arXiv API client
    client = arxiv.Client()

    # Construct the search query to look for the exact title
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=1
        # sort_by=arxiv.SortCriterion.Relevance
    )

    # Execute the search and retrieve results
    results = list(client.results(search))

    if results:
        # If a result is found, return the arXiv ID
        return results[0].entry_id.split('/')[-1]
    else:
        # If no results are found, return None
        return None

def Download_Latex_Repo(title, ProjectDir):
    arxivID = get_arxiv_id_by_title(title)
    if arxivID == None:
        return -1
    sourceUrl =  'https://arxiv.org/src/' + arxivID
    paper_path = os.path.join(ProjectDir, 'ReleventLiterature')
    if not os.path.exists(paper_path):
        os.makedirs(paper_path)
    SourcePath = os.path.join(paper_path, arxivID+".tar.gz")
    Extracted_path = os.path.join(paper_path, arxivID)
    if not os.path.exists(SourcePath):
        filename = wget.download(sourceUrl, out=SourcePath)
    with tarfile.open(SourcePath, 'r:gz') as tar:
        tar.extractall(path=Extracted_path)
    return arxivID

def search_google_scholar(title):
    # Note: Google Scholar scraping is discouraged and may be blocked
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Install BeautifulSoup4 for Google Scholar fallback: pip install beautifulsoup4")
        return None

    search_url = "https://scholar.google.com/scholar?q=" + urllib.parse.quote_plus(title)
    
    try:
        response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find first PDF link (this selector might change)
        pdf_link = soup.find('div', class_='gs_or_ggsm').a['href']
        return pdf_link
        # return pdf_link if pdf_link.endswith('.pdf') else None
        
    except Exception as e:
        print(f"Google Scholar search failed: {e}")
        return None

def sanitize_filename(filename: str) -> str:
    """
    Cleans up the filename by:
      - Replacing illegal characters with underscores.
      - Collapsing multiple whitespace characters into a single space.
      - Removing extra spaces around underscores.
    """
    # Replace characters that are not allowed in file names.
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # Collapse multiple whitespace characters into a single space.
    filename = re.sub(r'\s+', ' ', filename).strip()
    # Remove spaces around underscores (e.g., " _ " becomes "_")
    filename = re.sub(r'\s*_\s*', '_', filename)
    filename = filename.replace(' ', '_')  
    return filename

def download_pdf_by_title(title: str, download_dir: str) -> None:
    pdf_url = search_google_scholar(title)
    if pdf_url is None:
        return -1
    os.makedirs(download_dir, exist_ok=True)

    # Sanitize the paper title to create a valid filename
    filename = sanitize_filename(f"{title}.pdf")
    filepath = os.path.join(download_dir, filename)

    try:
        wget.download(pdf_url, out=filepath)
        # with open(filepath, "wb") as file:
        #     file.write(response.content)
        print("Downloaded PDF to %s", filepath)
        return filepath
    except IOError as e:
        print("Failed to save PDF file: %s", e)
        return -1
  
def extract_text_from_pdf(pdf_path):
    text = list()
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text())
    return text

def extract_sections_with_content(latex_code):
    """
    Extract sections and their content from LaTeX code.
    """
    # Pattern to match section headings
    section_pattern = re.compile(r'(\\(section|subsection|subsubsection)\*?\{([^}]*)\})')
    # List to hold sections
    sections = []
    # Current position in the LaTeX code
    pos = 0
    # Variable to hold the last found section
    last_section = None
    # Iterate over all section matches
    for match in section_pattern.finditer(latex_code):
        start, end = match.span()
        # Content between sections
        content = latex_code[pos:start]
        if last_section:
            last_section['content'] += content
            sections.append(last_section)
        # New section
        section_name = match.group(3)
        section_level = match.group(2)
        last_section = {'level': section_level, 'title': section_name, 'content': ''}
        pos = end
    # Add the content after the last section
    if last_section:
        last_section['content'] += latex_code[pos:]
        sections.append(last_section)
    else:
        # No sections found; treat entire document as one section
        content = latex_code[pos:]
        sections.append({'level': 'text', 'title': '', 'content': content})
    return sections

def resolve_latex_inputs(repo_path, latex_str, processed_files=None):
    """
    Recursively replaces all occurrences of \input{filename} in the LaTeX string with
    the content of the corresponding file found at repo_path.
    
    Parameters:
      repo_path (str): The root directory of your LaTeX repository.
      latex_str (str): The LaTeX content (string) which may contain \input{...} commands.
      processed_files (set): Internal set to track already processed files (to avoid circular loops).
    
    Returns:
      str: A new LaTeX string with all \input commands replaced by the file contents.
    """
    if processed_files is None:
        processed_files = set()
    
    # Pattern to match \input{...} (non-greedy, and allow spaces inside the braces)
    pattern = re.compile(r"\\input\{\s*([^}]+?)\s*\}")
    
    def replace_input(match):
        # Extract the filename from the match
        filename = match.group(1).strip()
        # If the filename does not have an extension, assume .tex
        if not os.path.splitext(filename)[1]:
            filename += ".tex"
        full_path = os.path.join(repo_path, filename)
        
        # Check for circular references
        if full_path in processed_files:
            return f"% [Warning: Circular reference detected for {full_path}]"
        
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            return f"% [Error: Could not open {full_path}: {e}]"
        
        # Mark this file as processed to avoid recursion loops.
        processed_files.add(full_path)
        # Recursively process the content of the input file.
        resolved_content = resolve_latex_inputs(repo_path, file_content, processed_files)
        return resolved_content
    
    # Substitute each \input command with the resolved file content.
    organized_str = re.sub(pattern, replace_input, latex_str)
    return organized_str

def extract_title(text):
    import re
    # Match pattern for title field allowing flexible spaces and either { or " as the opening delimiter
    match = re.search(r'title\s*=\s*([{\"])', text)
    if not match:
        return None
    token = match.group(1)
    start = match.end()  # Position after the opening token
    if token == '{':
        count = 1
        end = start
        # Parse to correctly account for nested braces
        while end < len(text) and count > 0:
            if text[end] == '{':
                count += 1
            elif text[end] == '}':
                count -= 1
            end += 1
        title_raw = text[start:end-1]
        title_clean = title_raw.replace('{', '').replace('}', '')
        return title_clean.strip()
    elif token == '"':
        end = text.find('"', start)
        if end == -1:
            return None
        title_raw = text[start:end]
        title_clean = title_raw.replace('{', '').replace('}', '')
        return title_clean.strip()

def read_bib_files(directory, recursive=True):
    latex_content = ''
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bib'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    latex_content += '\n' + content
    bib_list = latex_content.split('@')
    output = {}
    for i in range(0, len(bib_list)):
        index = bib_list[i].find('{')
        if index == -1:
            continue
        index1 = bib_list[i].find(',', index)
        label = bib_list[i][index+1:index1]
        title = extract_title(bib_list[i])
        output[label.strip()] = title
    return output
    
def read_latex_files(directory, recursive=True):
    """
    Read all .tex files in the given directory and return the concatenated content.
    """
    latex_content = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tex'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    latex_content.append(content)
                    # latex_content += '\n' + content
        if not recursive:
            break  # Do not traverse subdirectories
    return latex_content

def Extract_Latex_Section(latex_project_directory):
    latex_code = read_latex_files(latex_project_directory, recursive=True)
    sections = list()
    section_pattern = re.compile(r'(\\(section|subsection|subsubsection)\*?\{([^}]*)\})')
    for i in range(len(latex_code)):
        match = section_pattern.search(latex_code[i])
        if not match:
            continue 
        section = extract_sections_with_content(latex_code[i])
        sections.extend(section)
    
    match_pattern = re.compile(r"\\input\s*\{\s*[^}]+\s*\}")
    for i in range(len(sections)):
        sections[i]['retrieved'] = False
        sections[i]['content'] = clean_latex_content(sections[i]['content'])
        content = sections[i]['content']
        if match_pattern.search(content):
            sections[i]['content'] = resolve_latex_inputs(latex_project_directory, content)
    return sections

def clean_latex_content(content):
    """
    Clean LaTeX content by removing commands and special characters.
    """
    # Remove comments
    content = re.sub(r'%.*', '', content)
    return content.strip()
