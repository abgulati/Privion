# from kasa import SmartPlug as KasaSmartPlug
from kasa.iot import IotPlug as KasaSmartPlug

KASA_PLUG_IP_ADDR = "192.168.0.68"

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
    