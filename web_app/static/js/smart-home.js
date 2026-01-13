
let pollInterval = null;

function toggleSmartHome() {
    const smartHomeArea = document.getElementById('smart-home-area');
    if (smartHomeArea && smartHomeArea.style.display === 'block') {
        closeSmartHome();
    } else {
        openSmartHome();
    }
}


function openSmartHome() {
    // 1. Hide Chat Area and Input Area
    const chatArea = document.getElementById('chat-area');
    const inputArea = document.getElementById('input-area');
    const scrollBtn = document.getElementById('scrollDownButton');
    
    if (chatArea) chatArea.style.display = 'none';
    if (inputArea) inputArea.style.display = 'none';
    if (scrollBtn) scrollBtn.style.display = 'none';

    // 2. Show Smart Home Area
    let smartHomeArea = document.getElementById('smart-home-area');
    if (!smartHomeArea) {
        console.error("Smart Home Area not found in DOM");
        return;
    }
    smartHomeArea.style.display = 'block';

    // 3. Load Data & Start Polling
    loadSmartHomeDevices();
    startPolling();
    
    // 4. Close Sidenav
    closeNav();
}

function closeSmartHome() {
    document.getElementById('smart-home-area').style.display = 'none';
    
    const chatArea = document.getElementById('chat-area');
    const inputArea = document.getElementById('input-area');
    
    // Restore default display (assuming stylesheet handles it)
    if (chatArea) chatArea.style.display = ''; 
    if (inputArea) inputArea.style.display = '';

    stopPolling();
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(() => {
        loadSmartHomeDevices(true); // silent load
    }, 3000); // Poll every 3 seconds
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

async function loadSmartHomeDevices(silent = false) {
    const area = document.getElementById('smart-home-area');
    if (!silent) {
        area.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Loading Devices...</div>';
    }

    try {
        const response = await fetch('/api/smart_home/devices');
        const result = await response.json();

        if (result.success) {
            renderSmartHome(result.data);
        } else if (!silent) {
            area.innerHTML = `<div class="error-message">Error loading devices: ${result.error}</div>`;
        }
    } catch (e) {
        if (!silent) area.innerHTML = `<div class="error-message">Network error: ${e}</div>`;
    }
}

function renderSmartHome(data) {
    const area = document.getElementById('smart-home-area');
    area.innerHTML = '<div class="smart-home-header"><h2>Smart Home Control</h2><button class="btn btn-secondary btn-sm" onclick="closeSmartHome()">Back to Chat</button></div>';

    // Data is { Zone: { Room: [Devices] } }
    
    // Check if empty
    if (Object.keys(data).length === 0) {
        area.innerHTML += '<p>No devices found.</p>';
        return;
    }

    // Sort Zones
    const zones = Object.keys(data).sort();

    zones.forEach(zone => {
        const zoneDiv = document.createElement('div');
        zoneDiv.className = 'zone-section';
        
        const zoneTitle = document.createElement('div');
        zoneTitle.className = 'zone-title';
        zoneTitle.textContent = zone;
        zoneDiv.appendChild(zoneTitle);

        const rooms = data[zone];
        const roomKeys = Object.keys(rooms).sort();

        roomKeys.forEach(room => {
            const roomDiv = document.createElement('div');
            roomDiv.className = 'room-section';

            const roomTitle = document.createElement('div');
            roomTitle.className = 'room-title';
            roomTitle.textContent = room;
            roomDiv.appendChild(roomTitle);

            const deviceGrid = document.createElement('div');
            deviceGrid.className = 'device-grid';

            rooms[room].forEach(device => {
                const card = createDeviceCard(device);
                deviceGrid.appendChild(card);
            });

            roomDiv.appendChild(deviceGrid);
            zoneDiv.appendChild(roomDiv);
        });

        area.appendChild(zoneDiv);
    });
}

function createDeviceCard(device) {
    // Check if online and 'is_on' attribute
    const isOn = device.is_online && device.attributes && device.attributes.is_on;
    
    const card = document.createElement('div');
    card.className = `device-card ${isOn ? 'on' : 'off'}`;
    card.id = `device-card-${device.id}`;
    card.style.cursor = 'pointer'; // Make it look clickable

    // Click handler for total card
    card.onclick = (e) => {
        // Prevent toggle if clicking color picker or other inputs
        if (e.target.tagName === 'INPUT') return;
        
        const action = isOn ? 'turn_off' : 'turn_on';
        controlDevice(device.id, action);
    };

    // Icon based on type
    let iconClass = 'fa-question-circle';
    if (device.type === 'kasa_plug') iconClass = 'fa-plug';
    else if (device.type === 'kasa_bulb') iconClass = 'fa-lightbulb';
    
    // Icon Container
    const iconContainer = document.createElement('div');
    iconContainer.className = 'icon-container';
    
    const icon = document.createElement('i');
    icon.className = `fas ${iconClass} device-icon`;
    
    // Apply dynamic color if bulb and on
    if (device.type === 'kasa_bulb' && isOn) {
        if (device.attributes.color_temp && device.attributes.color_temp > 0) {
             // White Mode - Set icon to a representation of the white temperature
             // 2700K (Warm) -> Orange-ish
             // 5000K (Cool) -> Blue-ish White
             // Simple approximation:
             if (device.attributes.color_temp < 4000) {
                 icon.style.color = '#ffb74d'; // Warm Orange
             } else {
                 icon.style.color = '#ffffff'; // Pure White for "Bright White"
             }
        }
        else if (device.attributes.hsv) {
            // Color Mode (HSV)
            const [h, s, v] = device.attributes.hsv;
            icon.style.color = `hsl(${h}, ${s}%, 50%)`;
        }
    }
    
    iconContainer.appendChild(icon);
    card.appendChild(iconContainer);

    const name = document.createElement('div');
    name.className = 'device-name';
    name.textContent = device.name;
    name.title = device.name;
    card.appendChild(name);

    const ip = document.createElement('div');
    ip.className = 'device-ip';
    ip.textContent = device.ip_address || 'No IP';
    card.appendChild(ip);
    
    // --- Extra Controls for Bulb (Bubbles) ---
    if (device.type === 'kasa_bulb') {
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'bulb-controls';
        
        // 1. Bright White Preset (5000K)
        const whiteBubble = document.createElement('div');
        whiteBubble.className = 'color-bubble white-preset';
        whiteBubble.title = 'Bright White';
        whiteBubble.onclick = (e) => {
            e.stopPropagation();
            changeDeviceTemp(device.id, 5000);
        };
        controlsContainer.appendChild(whiteBubble);
        
        // 2. Warm White Preset (2700K)
        const warmBubble = document.createElement('div');
        warmBubble.className = 'color-bubble warm-preset';
        warmBubble.title = 'Warm White';
        warmBubble.onclick = (e) => {
            e.stopPropagation();
            changeDeviceTemp(device.id, 2700);
        };
        controlsContainer.appendChild(warmBubble);
        
        // 3. Custom Color Picker Bubble
        const customBubble = document.createElement('div');
        customBubble.className = 'color-bubble custom-preset';
        customBubble.title = 'Change Color';
        
        // Icon
        const dropper = document.createElement('i');
        dropper.className = 'fas fa-eye-dropper';
        customBubble.appendChild(dropper);
        
        // Color Input
        // Note: HTML input type="color" uses Hex (RGB). Kasa expects HSV.
        // We use hexToHsv() helper to bridge this.
        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        // Set initial value
        if (device.attributes && device.attributes.hsv) {
            const [h, s, v] = device.attributes.hsv;
            const hex = hsvToHex(h, s, v);
            colorInput.value = hex;
            customBubble.style.backgroundColor = hex;
            if (v < 50) dropper.style.color = '#fff';
            else dropper.style.color = '#333';
        } else {
             colorInput.value = '#ffffff';
             customBubble.style.backgroundColor = '#ffffff';
        }

        colorInput.onclick = (e) => {
             e.stopPropagation();
             stopPolling();
        };
        
        colorInput.onblur = () => {
             startPolling();
        };

        colorInput.onchange = (e) => {
            const hex = e.target.value;
            const hsv = hexToHsv(hex);
            changeDeviceColor(device.id, hsv);
            startPolling();
        };
        
        customBubble.appendChild(colorInput);
        controlsContainer.appendChild(customBubble);
        
        card.appendChild(controlsContainer);
    }

    return card;
}

// Helper: Hex to HSV
function hexToHsv(hex) {
  let r = 0, g = 0, b = 0;
  if (hex.length === 4) {
    r = "0x" + hex[1] + hex[1];
    g = "0x" + hex[2] + hex[2];
    b = "0x" + hex[3] + hex[3];
  } else if (hex.length === 7) {
    r = "0x" + hex[1] + hex[2];
    g = "0x" + hex[3] + hex[4];
    b = "0x" + hex[5] + hex[6];
  }
  r /= 255; g /= 255; b /= 255;
  let cmin = Math.min(r,g,b), cmax = Math.max(r,g,b), delta = cmax - cmin, h = 0, s = 0, v = 0;

  if (delta == 0) h = 0;
  else if (cmax == r) h = ((g - b) / delta) % 6;
  else if (cmax == g) h = (b - r) / delta + 2;
  else h = (r - g) / delta + 4;

  h = Math.round(h * 60);
  if (h < 0) h += 360;
  v = Math.round(cmax * 100);
  s = cmax == 0 ? 0 : Math.round((delta / cmax) * 100);

  return { h, s, v };
}

// Helper: HSV to Hex
function hsvToHex(h, s, v) {
  s /= 100; v /= 100;
  let c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c, r = 0, g = 0, b = 0;

  if (0 <= h && h < 60) { r = c; g = x; b = 0; }
  else if (60 <= h && h < 120) { r = x; g = c; b = 0; }
  else if (120 <= h && h < 180) { r = 0; g = c; b = x; }
  else if (180 <= h && h < 240) { r = 0; g = x; b = c; }
  else if (240 <= h && h < 300) { r = x; g = 0; b = c; }
  else if (300 <= h && h < 360) { r = c; g = 0; b = x; }

  r = Math.round((r + m) * 255).toString(16);
  g = Math.round((g + m) * 255).toString(16);
  b = Math.round((b + m) * 255).toString(16);

  if (r.length == 1) r = "0" + r;
  if (g.length == 1) g = "0" + g;
  if (b.length == 1) b = "0" + b;

  return "#" + r + g + b;
}

async function changeDeviceTemp(deviceId, temp) {
    try {
        const response = await fetch('/api/smart_home/control', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ device_id: deviceId, action: 'set_color_temp', temp: temp })
        });
        const result = await response.json();
        
        if (result.success) {
           loadSmartHomeDevices(true);
        } else {
            console.error(`Error setting color temp: ${result.error}`);
        }
    } catch (e) {
        console.error(`Network Error setting color temp: ${e}`);
    }
}

async function changeDeviceColor(deviceId, hsv) {
    try {
        const response = await fetch('/api/smart_home/control', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ device_id: deviceId, action: 'set_color', hsv: hsv })
        });
        const result = await response.json();
        
        if (result.success) {
           loadSmartHomeDevices(true);
        } else {
            console.error(`Error setting color: ${result.error}`);
        }
    } catch (e) {
        console.error(`Network Error setting color: ${e}`);
    }
}

async function controlDevice(deviceId, action) {
    try {
        const response = await fetch('/api/smart_home/control', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ device_id: deviceId, action: action })
        });
        const result = await response.json();
        
        if (result.success) {
           // Reload all devices to reflect state change locally and for others
           loadSmartHomeDevices();
        } else {
            alert(`Error: ${result.error || result.message}`);
        }
    } catch (e) {
        alert(`Network Error: ${e}`);
    }
}
