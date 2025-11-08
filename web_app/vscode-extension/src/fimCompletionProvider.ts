import * as vscode from 'vscode';

interface FIMRequest {
    prefix: string;
    suffix: string;
    middle?: string;
    language: string;
}

export class FIMCompletionProvider implements vscode.InlineCompletionItemProvider {
    private serverUrl!: string; // ! here is a defninite assignment assertion, and tells TypeScript "I promise these will be initialized before use"!
    private maxTokens!: number;
    private temperature!: number;
    private autoComplete!: boolean;
    private triggerDelay!: number;
    private pendingRequests = new Map<string, AbortController>();

    constructor() {
        // Initialize with default values
        this.updateConfig();

        // Listen for configuration changes
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('hfw-fim')) {
                this.updateConfig();    //
            }
        });
    }

    private updateConfig() {
        const config = vscode.workspace.getConfiguration('hfw-fim');
        this.serverUrl = config.get('serverUrl', 'http://localhost:9069/');
        this.maxTokens = config.get('maxTokens', 100);
        this.temperature = config.get('temperature', 0.3);
        this.autoComplete = config.get('autoComplete', true);
        this.triggerDelay = config.get('triggerDelay', 500);
    }

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {

        if (!this.autoComplete && context.triggerKind !== vscode.InlineCompletionTriggerKind.Invoke) {
            return [];
        }

        // Debounce requests
        const key = `${document.uri.toString()}-${position.line}-${position.character}`;
        if (this.pendingRequests.has(key)) {
            this.pendingRequests.get(key)?.abort();
        }

        const abortController = new AbortController();
        this.pendingRequests.set(key, abortController);

        try {
            // Small delay to avoid excessive requests
            await new Promise(resolve => setTimeout(resolve, this.triggerDelay));

            if (abortController.signal.aborted || token.isCancellationRequested) {
                return [];
            }

            const fimRequest = this.buildFIMRequest(document, position);
            if (!fimRequest) return [];

            const completion = await this.fetchCompletion(fimRequest, abortController.signal);
            if (!completion || abortController.signal.aborted) {
                return [];
            }

            return [new vscode.InlineCompletionItem(completion)];

        } catch (error) {
            if (error instanceof Error && error.name !== 'AbortError') {
                console.error('FIM completion error:', error);
            }
            return [];
        } finally {
            this.pendingRequests.delete(key);
        }
    }

    private buildFIMRequest(document: vscode.TextDocument, position: vscode.Position): FIMRequest | null {
        const offset = document.offsetAt(position);
        const text = document.getText();

        const prefix = text.substring(0, offset);
        const suffix = text.substring(offset);

        // Skip if cursor is at the very beginning or end
        if (!prefix.trim() && !suffix.trim()) {
            return null;
        }

        const language = this.getLanguageId(document.languageId);

        return { prefix, suffix, language };
    }

    private getLanguageId(vscodeLangId: string): string {
    const mapping: { [key: string]: string } = {
        // JavaScript variants
        'javascript': 'javascript',
        'javascriptreact': 'javascript',
        'jsx': 'javascript',
        
        // TypeScript variants
        'typescript': 'typescript',
        'typescriptreact': 'typescript',
        'tsx': 'typescript',
        
        // C family
        'c': 'c',
        'cpp': 'cpp',
        'c++': 'cpp',
        'objective-c': 'c',
        'objective-cpp': 'cpp',
        
        // Java
        'java': 'java',
        
        // Python
        'python': 'python',
        
        // Rust
        'rust': 'rust',
        
        // Go
        'go': 'go',
        
        // PHP
        'php': 'php',
        
        // Ruby
        'ruby': 'ruby',
        
        // Swift
        'swift': 'swift',
        
        // Kotlin
        'kotlin': 'kotlin',
        
        // Scala
        'scala': 'scala',
        
        // Web languages
        'html': 'html',
        'css': 'css',
        'scss': 'css',
        'sass': 'css',
        'less': 'css',
        
        // Data formats
        'json': 'json',
        'jsonc': 'json',
        'yaml': 'yaml',
        'yml': 'yaml',
        'xml': 'xml',
        'markdown': 'markdown',
        'md': 'markdown',
        
        // SQL
        'sql': 'sql',
        'mysql': 'sql',
        'postgresql': 'sql',
        
        // Shell
        'shellscript': 'bash',
        'bash': 'bash',
        'sh': 'bash',
        'zsh': 'bash',
        
        // PowerShell
        'powershell': 'powershell',
        'ps1': 'powershell',
        
        // Docker
        'dockerfile': 'bash',
        'dockercompose': 'yaml'
    };
    
    return mapping[vscodeLangId] || vscodeLangId;
    }

    private async fetchCompletion(request: FIMRequest, signal: AbortSignal): Promise<string | null> {
        const http = require('http');  // Use http instead of https
        
        return new Promise((resolve) => {
            const postData = JSON.stringify(request);
            const options = {
                hostname: 'localhost',
                port: 9069,
                path: '/exl2_fim_stream',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Max-New-Tokens': this.maxTokens.toString(),
                    'X-Temperature': this.temperature.toString(),
                    'Content-Length': Buffer.byteLength(postData)
                }
            };

            const req = http.request(options, (res: any) => {
                let data = '';
                res.on('data', (chunk: any) => data += chunk);
                res.on('end', () => {
                    // Parse response
                    const lines = data.split('\n');
                    let completion = '';
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const content = line.slice(6).replace(/^"|"$/g, '');
                            if (content !== 'null' && content !== '[DONE]') {
                                completion += content;
                            }
                        }
                    }
                    resolve(completion.trim() || null);
                });
            });

            req.on('error', (error: any) => {
                console.error('API Request Error:', error);
                resolve(null);
            });

            signal.addEventListener('abort', () => req.destroy());
            req.write(postData);
            req.end();
        });
    }

    async triggerCompletion(editor: vscode.TextEditor) {
        const position = editor.selection.active;
        const document = editor.document;

        const fimRequest = this.buildFIMRequest(document, position);
        if (!fimRequest) return;

        const completion = await this.fetchCompletion(fimRequest, new AbortController().signal);
        if (!completion) return;

        await editor.edit(editBuilder => {
            editBuilder.insert(position, completion);
        });
    }

    async testAPIConnection(): Promise<boolean> {
        const http = require('http');  // Use http, not https
        
        return new Promise((resolve) => {
            const postData = JSON.stringify({
                prefix: "def test():",
                suffix: "",
                language: "python"
            });
            
            const options = {
                hostname: 'localhost',
                port: 9069,
                path: '/exl2_fim_stream',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(postData)
                }
            };
            
            const req = http.request(options, (res: any) => {
                console.log('API Response Status:', res.statusCode);
                resolve(res.statusCode === 200);
            });
            
            req.on('error', (error: any) => {
                console.error('API Connection Error:', error);
                resolve(false);
            });
            
            req.setTimeout(5000, () => {  // Increased timeout
                console.error('API Timeout');
                req.destroy();
                resolve(false);
            });
            
            req.write(postData);
            req.end();
        });
    }
}