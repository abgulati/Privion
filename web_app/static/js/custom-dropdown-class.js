class CustomDropdown {
    constructor(containerElement, options = {}) {
        this.container = containerElement;
        this.options = options;

        // Read configuration from data attributes
        this.type = containerElement.dataset.type;
        this.configSource = containerElement.dataset.configSource;
        this.listKey = containerElement.dataset.listKey;
        this.selectedKey = containerElement.dataset.selectedKey;
        this.disabled = containerElement.dataset.disabled;

        // Validate required attributes
        if (!this.configSource || !this.listKey || !this.selectedKey) {
            console.error('CustomDropdown: Missing required data attributes', {
                configSource: this.configSource,
                listKey: this.listKey,
                selectedKey: this.selectedKey
            });
            return;
        }

        // Initialize the dropdown
        this.cacheElements();   // Cache DOM elements
        this.bindEvents();  // Bind events
        this.loadAndRender();   // load initial data

        // Store the created object so it can be referenced elsewhere in JS (eg save-config!s)
        if (!CustomDropdown.registry) CustomDropdown.registry = new Map();
        CustomDropdown.registry.set(this.type, this);
    }
    

    /**
     * Cache references to all child elements
     */
    cacheElements() {
        this.header = this.container.querySelector('.custom-dropdown-header');
        this.selectedSpan = this.header.querySelector('.custom-dropdown-selected');
        this.arrow = this.header.querySelector('.custom-dropdown-arrow');
        this.content = this.container.querySelector('.custom-dropdown-content');
        this.sortBtn = this.container.querySelector('.custom-dropdown-sort-btn');
        this.searchInput = this.container.querySelector('.custom-dropdown-search');
        this.addBtn = this.container.querySelector('.custom-dropdown-add-btn');
        this.addInput = this.container.querySelector('.custom-dropdown-add-input');
        this.itemsList = this.container.querySelector('.custom-dropdown-items-list');
    }


    /**
     * Bind all event listeners
     */
    bindEvents() {
        // Toggle dropdown open/close
        // Need to store a ref this handler must be removable in disable() and JS creates a new
        // anonymous function object everytime `() => this.toggleDropdown()` runs!
        this._toggleHandler = () => this.toggleDropdown();
        this.header.addEventListener('click', this._toggleHandler);

        // Search/filter
        this.searchInput.addEventListener('input', (e) => this.filterItems(e.target.value));

        // Sort
        this.sortBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.reverseSort();
        });

        // Add new item
        this.addBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleAddInput();
        });

        // Add input - handle Enter key
        this.addInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const value = this.addInput.value.trim();
                if (value) {
                    this.addItem(value);
                    this.addInput.value = '';
                    this.toggleAddInput();  // Hide input, show button
                }
            }
        });

        // Close when clicking outside
        // document.addEventListener('click', (e) => {
        //     if (!this.container.contains(e.target)) {
        //         this.closeDropdown();
        //     }
        // });

    }


    toggleDropdown() {
        this.content.classList.toggle('show');
    }


    closeDropdown() {
        this.content.classList.remove('show');
    }


    openDropdown() {
        this.content.classList.add('show');
    }


    enable() {
        this.header.style.cursor = 'pointer';
        this.header.style.backgroundColor = 'white';
        this.header.addEventListener('click', this._toggleHandler);
    }


    disable() {
        this.header.style.cursor = 'not-allowed';
        this.header.style.backgroundColor = 'darkgray';
        this.header.removeEventListener('click', this._toggleHandler);
    }


    toggleAddInput() {
        const wasBtnVisible = this.addBtn.style.display === 'block';
        this.addBtn.style.display = wasBtnVisible ? 'none' : 'block';
        this.addInput.style.display = wasBtnVisible ? 'block' : 'none';

        if (wasBtnVisible) {    // Btn was visible but now the input field is
            this.addInput.focus();
        }
    }


    filterItems(searchWords) {
        const items = this.itemsList.querySelectorAll('.custom-dropdown-item');
        const query = searchWords.toLowerCase();

        items.forEach(item => {
            const text = item.querySelector('.custom-dropdown-item-text').textContent.toLowerCase();
            item.style.display = text.includes(query) ? '' : 'none';
        });
    }


    selectItem(item) {
        this.selectedSpan.textContent = item;
        this.closeDropdown();

        const items = this.itemsList.querySelectorAll('.custom-dropdown-item');
        items.forEach(el => {
            const text = el.querySelector('.custom-dropdown-item-text').textContent;
            el.classList.toggle('selected', text.toLowerCase() === item.toLowerCase());
        });

        if (this.options.onItemSelect) {
            this.options.onItemSelect(item);
        }   // optional callback
    }


    renderItems(list, selectedItem) {
        this.itemsList.innerHTML = '';

        if (selectedItem) {
            this.selectedSpan.textContent = selectedItem;
        }

        list.forEach(item => {
            const div = document.createElement('div');
            div.className = 'custom-dropdown-item';
            div.innerHTML = `
                <span class="custom-dropdown-item-text">${item}</span>
                <span class="custom-dropdown-delete-btn" title="Remove">×</span>
            `;

            // Highlight if selected
            if (item.toLowerCase() === (selectedItem || '').toLowerCase()) {
                div.classList.add('selected');
            }
            
            const deleteButton = div.querySelector('.custom-dropdown-delete-btn');
            deleteButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this.removeItem(item);
            });
            
            div.addEventListener('click', () => { this.selectItem(item) });
            
            this.itemsList.appendChild(div);
        });
    }


    loadConfig() {
        const loaders = {
            'hf': loadCoreHfConfig, // func refs
            'lars': loadCoreLarsConfig,
        };

        const loader = loaders[this.configSource];
        if (!loader) {
            throw new Error(`Unknown config source: ${this.configSource}`);
        }

        return loader();    // func call
    }


    async loadAndRender() {
        try {
            const config = await this.loadConfig();
            const list = config[this.listKey] || [];
            const selected = config[this.selectedKey] || '';
            this.renderItems(list, selected);

            if (this.options.onLoad) {
                this.options.onLoad(this);  // call with `options.onLoad = (instance) => { ...`
            }   // optional callback

            if (this.disabled === 'true') {
                this.disable();
            }

        } catch (error) {
            errorHandler(
                `loading dropdown data for ${this.listKey}`,
                'CustomDropdown.loadAndRender()',
                String(error.message)
            );
        }
    }


    async updateList(newList) {
        if (this.configSource === 'hf') {
            await updateHfModelList(newList);
        } else if (this.configSource === 'lars') {
            await updateLarsCustomModelList(this.listKey, newList);
        }
    }


    async addItem(item) {
        try {
            const config = await this.loadConfig();
            const selected = config[this.selectedKey] || '';
            const currentList = config[this.listKey] || [];
            const newList = [...currentList, item];
            
            await this.updateList(newList);

            // If additional action callbacks are defined, call them now
            if (this.options.onItemAdd) {
                await this.options.onItemAdd(item);
            }

            this.renderItems(newList, selected);
            // TODO - select newly added?
        } catch (error) {
            errorHandler(
                `adding item from ${this.listKey} dropdown`,
                'CustomDropdown.addItem()',
                String(error.message)
            );
        }
    }


    async removeItem(item) {
        try {
            const config = await this.loadConfig();
            const selected = config[this.selectedKey] || '';
            const currentList = config[this.listKey] || [];
            const filteredList = currentList.filter(m => m.toLowerCase() !== item.toLowerCase());
            
            await this.updateList(filteredList);

            // If additional action callbacks are defined, call them now
            if (this.options.onItemRemove) {
                await this.options.onItemRemove(item);
            }

            this.renderItems(filteredList, selected);
        } catch (error) {
            errorHandler(
                `removing item from ${this.listKey} dropdown`,
                'CustomDropdown.removeItem()',
                String(error.message)
            );
        }
    }


    async reverseSort() {
        try {
            const config = await this.loadConfig();
            const selected = config[this.selectedKey] || '';
            const currentList = config[this.listKey] || [];
            const reversedList = [...currentList].reverse(); // creates & mutates a shallow copy

            await this.updateList(reversedList);

            this.renderItems(reversedList, selected);
        } catch (error) {
            errorHandler(
                `sorting ${this.listKey} dropdown`,
                'CustomDropdown.reverseSort()',
                String(error.message)
            );
        }
    }


    getSelectedValue() {
        return this.selectedSpan.textContent;
    }


    setSelectedValue(value) {
        this.selectItem(value);
    }


    refresh() {
        return this.loadAndRender();
    }
}


// cleaner to have a single deligated listener for the below rather than adding
// a global event target everytime initializeAllCustomDropdowns() is called!
document.addEventListener('click', (e) => {
    if (!CustomDropdown.registry) return;
    CustomDropdown.registry.forEach(dropdown => {
        if (!dropdown.container.contains(e.target)) {
            dropdown.closeDropdown();
        }
    });
});


window.CustomDropdown = CustomDropdown;