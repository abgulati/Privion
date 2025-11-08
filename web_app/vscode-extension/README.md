# HF-Waitress FIM Code Completion Extension

AI-powered Fill-in-the-Middle code completion using ExLlamaV2 models via the HF-Waitress LLM Server's `/exl2_fim_stream` API.

## Features

- **Streaming FIM completions** - Get intelligent code suggestions as you type
- **Multi-language support** - Works with 20+ programming languages
- **Real-time generation** - Uses streaming API for responsive completions
- **Configurable parameters** - Adjust temperature, max tokens, and more
- **Auto-completion toggle** - Enable/disable automatic suggestions

## Requirements

- HF-Waitress LLM Server Running Locally (default: http://localhost:9069)
- VSCode 1.74.0 or higher

## Configuration

Access settings via `Ctrl+,` → Search for "HF-Waitress FIM":

- `hfw-fim.serverUrl`: Your HF-Waitress URL
- `hfw-fim.maxTokens`: Maximum tokens to generate (default: 100)
- `hfw-fim.temperature`: Sampling temperature (default: 0.3)
- `hfw-fim.autoComplete`: Enable automatic completions (default: true)
- `hfw-fim.triggerDelay`: Delay before triggering (default: 500ms)

## Usage

1. **Automatic completions**: Just type in supported files
2. **Manual trigger**: Press `Ctrl+Space` or use "Generate FIM Completion" command
3. **Toggle auto-complete**: Use "Toggle Auto Completion" command

## Commands

- `HF-Waitress FIM: Generate Completion` - Trigger completion manually
- `HF-Waitress FIM: Toggle Auto Completion` - Enable/disable automatic suggestions

## Supported Languages

Python, JavaScript, TypeScript, Java, C/C++, Rust, Go, PHP, Ruby, Swift, Kotlin, Scala, HTML, CSS, SQL, Bash, PowerShell, YAML, JSON, XML, Markdown

## Installation instructions

Simply use the pre-packaged VSIX installer:

1. Open VSCode
2. Hit `Ctrl+Shift+P`
3. Type `Extension: Install from VSIX` and proceed to install

## For Developers

If making changes to the extensions code, recompile and reinstall:

1. Initialize the extension:
    ```
    npm init -y
    npm install --save-dev @types/vscode @types/node typescript
    ```

2. Compile:
    ```
    npm run compile
    ```

3. Install VSCode Extension CLI:
    ```
    npm install -g vsce
    ```

4. Package:
    ```
    vsce package
    ```

5. Install as per Installation Instructions above
