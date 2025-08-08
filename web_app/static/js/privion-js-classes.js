class ToolChainTraceManager {
    constructor(userMessageElement) {
        this.userMessageElement = userMessageElement;
        this.traceContainer = userMessageElement.querySelector('.tool-chain-trace');
        this.currentTimer = null;
        this.currentTraceItem = null; // We can now safely store the element reference
        this.startTime = null;
    }

    // Makes the trace container visible if it's not already.
    showTrace() {
        if (this.traceContainer && !this.traceContainer.classList.contains('expanded')) {
            this.traceContainer.classList.add('expanded');
            const toggleIcon = this.userMessageElement.querySelector('.tool-chain-toggle');
            if (toggleIcon) {
                toggleIcon.classList.add('expanded');
            }
        }
    }

    // Makes the trace container invisible if it's not already.
    hideTrace() {
        if (this.traceContainer && this.traceContainer.classList.contains('expanded')) {
            this.traceContainer.classList.remove('expanded');
            const toggleIcon = this.userMessageElement.querySelector('.tool-chain-toggle');
            if (toggleIcon) {
                toggleIcon.classList.remove('expanded');
            }
        }
    }

    // Starts a new step.
    startStep(stepDescription) {
        this.showTrace();
        this.stopCurrentTimer();

        // If a previous step was active, mark it as completed.
        if (this.currentTraceItem) {
            this.currentTraceItem.classList.remove('active');
            this.currentTraceItem.classList.add('completed');
        }

        let traceItem = this.findTraceItemByDescription(stepDescription);
        if (!traceItem) {
            traceItem = this.addNewTraceItem(stepDescription);
        }

        this.currentTraceItem = traceItem;
        this.currentTraceItem.className = 'tool-chain-trace-item active'; // Set class to active

        this.startTime = Date.now();
        const timerSpan = this.currentTraceItem.querySelector('.timer');
        if (!timerSpan) return;

        timerSpan.textContent = '0.0s';

        this.currentTimer = setInterval(() => {
            // Check if the item still exists and is the active one before updating
            if (this.currentTraceItem && this.currentTraceItem.classList.contains('active')) {
                const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
                timerSpan.textContent = `${elapsed}s`;
            } else {
                this.stopCurrentTimer();
            }
        }, 100);
    }

    // Completes the final step.
    completeCurrentStep() {
        this.hideTrace();
        if (this.currentTraceItem) {
            const timerSpan = this.currentTraceItem.querySelector('.timer');
            if (timerSpan && this.startTime) {
                const finalElapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
                timerSpan.textContent = `${finalElapsed}s`;
            }
            this.currentTraceItem.classList.remove('active');
            this.currentTraceItem.classList.add('completed');
            this.currentTraceItem = null; // Clear the reference to the last active item
        }
        this.stopCurrentTimer();
    }

    // Adds a new trace item div to the container.
    addNewTraceItem(description) {
        if (!this.traceContainer) return null;

        const traceItem = document.createElement('div');
        traceItem.className = 'tool-chain-trace-item';
        traceItem.innerHTML = `
            <span>${description}</span>
            <span class="timer">0.0s</span>
        `;
        this.traceContainer.appendChild(traceItem);
        return traceItem;
    }

    // Finds an existing trace item by its text content.
    findTraceItemByDescription(description) {
        if (!this.traceContainer) return null;

        const items = this.traceContainer.querySelectorAll('.tool-chain-trace-item');
        for (const item of items) {
            const span = item.querySelector('span:first-child');
            if (span && span.textContent.trim() === description) {
                return item;
            }
        }
        return null;
    }

    // Stops and clears the current timer interval.
    stopCurrentTimer() {
        if (this.currentTimer) {
            clearInterval(this.currentTimer);
            this.currentTimer = null;
        }
    }

    // Cleans up resources.
    destroy() {
        this.stopCurrentTimer();
        this.currentTraceItem = null;
    }
}