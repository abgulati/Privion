import json
from pathlib import Path
from aiowebostv.webos_client import WebOsClient

KEYS_FILE = Path("ha_device_modules/butler_webos_keys.json")

def _load_client_key(host: str) -> str | None:
    if not KEYS_FILE.exists():
        return None
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        return data.get(host)
    except Exception:
        return None

def _save_client_key(host: str, client_key: str) -> None:
    data = {}
    if KEYS_FILE.exists():
        try:
            data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[host] = client_key
    KEYS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

async def webos_pair_connect_and_power_off_async(host: str) -> dict:
    client_key = _load_client_key(host)
    client = WebOsClient(host, client_key=client_key)
    try:
        connected = await client.connect()
        if not connected:
            return {"success": False, "message": "Failed to connect to TV."}

        # If this was first-time pairing, accept the prompt on the TV.
        # After acceptance, the client_key will be set—persist it.
        if not client_key and client.client_key:
            _save_client_key(host, client.client_key)

        await client.power_off()
        return {"success": True, "message": "Power-off command sent."}
    except Exception as e:
        return {"success": False, "message": f"Error: {e}"}
    finally:
        await client.disconnect()