import * as vscode from 'vscode';
import { FIMCompletionProvider } from './fimCompletionProvider';

export function activate(context: vscode.ExtensionContext) {
    console.log('🚀 HF-Waitress FIM Extension activated!');

    // Force activation notification
    vscode.window.showInformationMessage('HF-Waitres FIM Extension Loaded');

    const testCommand = vscode.commands.registerCommand('hfw-fim.test', () => {
        vscode.window.showInformationMessage('HF-Waitress FIM Extension is working!');
    });

    const provider = new FIMCompletionProvider();
    provider.testAPIConnection().then(connected => {
        console.log('API Connected:', connected);
        if (!connected) {
            vscode.window.showWarningMessage('HF-Waitress API not found at localhost:9069');
        } else {
            vscode.window.showInformationMessage('HF-Waitres FIM Extension Loaded');
        }
    });

    // Register completion provider for multiple languages
    const languages = [
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'rust', 
        'go', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'html', 'css', 
        'sql', 'bash', 'powershell', 'yaml', 'json', 'xml', 'markdown'
    ];

    const disposables = languages.map(lang =>
        vscode.languages.registerInlineCompletionItemProvider(
            { scheme: 'file', language: lang },
            provider
        )
    );

    // Manual completion command
    const manualCompletion = vscode.commands.registerCommand(
        'hfw-fim.generateCompletion',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            await provider.triggerCompletion(editor);
        }
    );

    // Toggle auto-completion command
    const toggleAutoComplete = vscode.commands.registerCommand(
        'hfw-fim.toggleAutoComplete',
        () => {
            const config = vscode.workspace.getConfiguration('hfw-fim');
            const current = config.get('autoComplete', true);
            config.update('autoComplete', !current, vscode.ConfigurationTarget.Global);
            vscode.window.showInformationMessage(
                `HF-Waitress FIM Auto-Completion ${!current ? 'enabled' : 'disabled'}.`
            );
        }
    );

    context.subscriptions.push(...disposables, manualCompletion, toggleAutoComplete);
}

export function deactivate() {} // deactivate must be present because of VSCode extension requirements, but nothing to clean up here.