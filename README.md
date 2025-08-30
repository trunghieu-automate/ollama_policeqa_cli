# `policeqa` - Prompt Refinement and Execution CLI Tool

A command-line tool for refining prompts and then execute it using local Ollama models (gemma3:4b, qwen3:4b, phi4-mini:3.8b).

## Installation

1. Install Ollama: https://ollama.com/
2. Install required packages:
```bash
pip install ollama
```

3. Download the scripts and setup
```bash
wget https://raw.githubusercontent.com/trunghieu-automate/ollama_policeqa_cli/refs/heads/main/policeqa_cli.py
chmod +x policeqa_cli.py
sudo ln -s $(pwd)/policeqa_cli.py /usr/local/bin/policeqa
```

For mac user
```
echo "alias policeqa='python3 <policeqa_cli.py location>'" >> ~/.zshrc
source ~/.zshrc
```
## Usage
### Basic Prompt Refinement
```bash
policeqa --model gemma3:4b --prompt "tell me a story about a robot"
```

### Batch Processing
```bash
policeqa --model qwen3:4b --file prompts.txt
```

### Interactive Mode
```bash
policeqa
```

### Session management
```bash
# List all sessions
policeqa --list

# View a specific session
policeqa --session session_1712345678

# Save results to file
policeqa --model phi4-mini:3.8b --prompt "explain quantum computing" --output refined.json
```


## File Format
For batch processing, create a text file with one prompt per line:
```txt
Write a poem about the ocean
Explain quantum computing in simple terms
Create a recipe for chocolate cake
```

## Session Storage
All sessions are saved in prompt_sessions/ directory as JSON files containing:

- Initial prompts
- Q&A refinement history
- Final refined system and user prompts
- Model used and timestamps

## Requirements
- Python 3.7+
- Ollama running locally
- One of the supported models: gemma3:4b, qwen3:4b, phi4-mini:3.8b
