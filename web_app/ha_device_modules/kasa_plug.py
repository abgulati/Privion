# from kasa import SmartPlug as KasaSmartPlug
from kasa.iot import IotPlug as KasaSmartPlug
from kasa import Discover

KASA_PLUG_IP_ADDR = "192.168.0.68"

async def find_device_ip(mac_address: str) -> str | None:
    # Broadcast discovery
    found = await Discover.discover(target="255.255.255.255")
    target_mac = mac_address.replace(':', '').upper()
    
    for dev in found.values():
        dev_mac = dev.mac.replace(':', '').upper()
        if dev_mac == target_mac:
            # If the mac address of found device matches the target mac address, 
            # return the ip address of the found device
            return dev.host
    return None

async def check_online_handler(ip_address: str) -> dict:
    dev = KasaSmartPlug(ip_address)
    try:
        await dev.update()
        await dev.protocol.close()  # Fix: Ensure socket is closed
        return {"success": True, "message": "online"}
    except Exception as e:
        try: 
            await dev.protocol.close() 
        except: 
            pass
        return {"success": False, "message": f"offline: {e}"}

async def kasa_connect():
    try:
        dev = KasaSmartPlug(KASA_PLUG_IP_ADDR)
        await dev.update()
        return dev
    except Exception as e:
        return {"success": False, "message": f"Error connecting to Kasa plug: {e}"}
    
async def kasa_disconnect(dev):
    try:
        await dev.protocol.close()
        dev = None
    except Exception as e:
        return {"success": False, "message": f"Error disconnecting from Kasa plug: {e}"}

async def plug_turn_on_handler():
    try:
        dev = await kasa_connect()
        await dev.turn_on()
        print("Turned ON")
        await kasa_disconnect(dev)
        return {"success": True, "message": "Turn-on command sent to Kasa plug."}
    except Exception as e:
        return {"success": False, "message": f"Error turning on Kasa plug: {e}"}

async def plug_turn_off_handler():
    try:
        dev = await kasa_connect()
        await dev.turn_off()
        print("Turned OFF")
        await kasa_disconnect(dev)
        return {"success": True, "message": "Turn-off command sent to Kasa plug."}
    except Exception as e:
        return {"success": False, "message": f"Error turning off Kasa plug: {e}"}
    