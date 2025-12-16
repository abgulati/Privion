from kasa.iot import IotBulb as KasaSmartBulb

KASA_BULB_IP_ADDR = "192.168.0.164"

async def kasa_connect():
    try:
        dev = KasaSmartBulb(KASA_BULB_IP_ADDR)
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

async def bulb_turn_on_handler():
    try:
        dev = await kasa_connect()
        await dev.turn_on()
        print("Turned ON")
        await kasa_disconnect(dev)
        return {"success": True, "message": "Turn-on command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error turning on Kasa bulb: {e}"}

async def bulb_turn_off_handler():
    try:
        dev = await kasa_connect()
        await dev.turn_off()
        print("Turned OFF")
        await kasa_disconnect(dev)
        return {"success": True, "message": "Turn-off command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error turning off Kasa bulb: {e}"}
    
async def bulb_set_brightness_handler(brightness: int):
    try:
        dev = await kasa_connect()
        await dev.modules["Light"].set_brightness(brightness)
        return {"success": True, "message": "Brightness set command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error setting the brightness on Kasa bulb: {e}"}

async def bulb_set_color_handler(hue: int, saturation: int, value: int):
    try:
        dev = await kasa_connect()
        await dev.modules["Light"].set_hsv(hue, saturation, value)
        return {"success": True, "message": "Color set command sent to Kasa bulb."}
    except Exception as e:
        return {"success": False, "message": f"Error setting the color on Kasa bulb: {e}"}
    