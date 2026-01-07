from kasa.iot import IotBulb as KasaSmartBulb
from kasa import Discover

async def find_device_ip(mac_address: str) -> str | None:
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
    dev = KasaSmartBulb(ip_address)
    try:
        await dev.update()
        
        # Collect state attributes
        state = {
            "is_on": dev.is_on,
            "brightness": dev.brightness,
            "hsv": dev.hsv,
            "color_temp": dev.color_temp,
            # Add any other relevant attributes here
        }
        
        await dev.protocol.close()  # Fix: Ensure socket is closed
        return {"success": True, "message": "online", "state": state}
    except Exception as e:
        try:
            await dev.protocol.close()
        except:
            pass
        return {"success": False, "message": f"offline: {e}"}

async def kasa_connect(ip_address: str):
    try:
        dev = KasaSmartBulb(ip_address)
        await dev.update()
        return dev
    except Exception as e:
        return {"success": False, "message": f"Error connecting to Kasa bulb: {e}"}
    
async def kasa_disconnect(dev):
    try:
        await dev.protocol.close()
        dev = None
    except Exception as e:
        return {"success": False, "message": f"Error disconnecting from Kasa bulb: {e}"}

async def bulb_turn_on_handler(ip_address: str):
    try:
        dev = await kasa_connect(ip_address)
        await dev.turn_on()
        print("Turned ON")
        await kasa_disconnect(dev)
        return {"success": True, "message": "Turn-on command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error turning on Kasa bulb: {e}"}

async def bulb_turn_off_handler(ip_address: str):
    try:
        dev = await kasa_connect(ip_address)
        await dev.turn_off()
        print("Turned OFF")
        await kasa_disconnect(dev)
        return {"success": True, "message": "Turn-off command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error turning off Kasa bulb: {e}"}
    
async def bulb_set_brightness_handler(ip_address: str, brightness: int):
    try:
        dev = await kasa_connect(ip_address)
        await dev.modules["Light"].set_brightness(brightness)
        await kasa_disconnect(dev) # Not required but best practice to disconnect so we can free the device up for other requests
        return {"success": True, "message": "Brightness set command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error setting the brightness on Kasa bulb: {e}"}

async def bulb_set_color_handler(ip_address: str, hue: int, saturation: int, value: int):
    try:
        dev = await kasa_connect(ip_address)
        await dev.modules["Light"].set_hsv(hue, saturation, value)
        await kasa_disconnect(dev) # Not required but best practice to disconnect so we can free the device up for other requests
        return {"success": True, "message": "Color set command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error setting the color on Kasa bulb: {e}"}
    