import sys
import asyncio
import socket
from contextlib import suppress

from privion_config_concierge import read_config
from govee_local_api import GoveeController

# Windows fixes
if sys.platform == "win32":
    # Prefer Selector loop for UDP/multicast stability on Windows - Avoids some Windows UDP oddities
    with suppress(Exception):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # reuse_port is not supported on Windows
    with suppress(Exception):
        import asyncio.base_events as _base_events
        _base_events._set_reuseport = lambda sock: None  # no-op


# Ignore multicast setsockopt error (WinError 10022) during connection_made
def _patch_govee_on_windows():
    if sys.platform != "win32":
        return
    orig = getattr(GoveeController, "connection_made", None)
    if not callable(orig):
        return

    def patched(self, transport):
        try:
            return orig(self, transport)
        except OSError as e:
            if getattr(e, "winerror", None) == 10022:
                print("Ignoring WinError 10022 during multicast setup; continuing without multicast join.")
                # Discovery uses unicast replies to listening socket; missing the multicast join only affects passive announcements, not active discovery.
                return
            raise

    GoveeController.connection_made = patched


_patch_govee_on_windows()


BROADCAST_ADDR = "239.255.255.250"
TARGET_PORT = 4001
LISTENING_PORT = 4002
DISCOVERY_TIMEOUT = 5
GOVEE_CONTROLLER = None


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with suppress(Exception):
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    s.close()
    return "0.0.0.0"


async def discover_controller(local_ip: str) -> GoveeController:
    loop = asyncio.get_running_loop()

    global GOVEE_CONTROLLER

    if GOVEE_CONTROLLER is None:
        GOVEE_CONTROLLER = GoveeController(
            loop=loop,
            listening_address=local_ip,
            broadcast_address=BROADCAST_ADDR,
            broadcast_port=TARGET_PORT,
            listening_port=LISTENING_PORT,
            discovery_enabled=True,
            discovery_interval=1,
            update_enabled=False,
        )
        await GOVEE_CONTROLLER.start()
    
    # Wait up to DISCOVERY_TIMEOUT seconds for at least one device
    for _ in range(DISCOVERY_TIMEOUT * 2):
        if GOVEE_CONTROLLER.devices:
            break
        await asyncio.sleep(0.5)
    return GOVEE_CONTROLLER


def pick_device(ctrl: GoveeController, ident: str | None):
    if not ctrl.devices:
        return None
    if not ident:
        return ctrl.devices[0]
    ident_l = ident.lower()
    for d in ctrl.devices:
        sku = getattr(d, "sku", "") or ""
        fp = getattr(d, "fingerprint", "") or ""
        if ident_l in sku.lower() or ident_l in fp.lower():
            return d
    return None


async def _cleanup_controller(ctrl: GoveeController) -> None:
    """Best-effort cleanup regardless of return shape."""
    result = ctrl.cleanup()
    events = result if isinstance(result, list) else [result]
    waiters = []
    for ev in events:
        if hasattr(ev, "wait"):
            waiters.append(asyncio.wait_for(ev.wait(), 1))
    if waiters:
        with suppress(asyncio.TimeoutError):
            await asyncio.gather(*waiters)


async def get_first_govee_device():

    local_ip = get_local_ip()
    print(f"Using IP: {local_ip}")

    ctrl = None
    try:
        ctrl = await discover_controller(local_ip)
    except OSError as ex:
        print(f"Failed to start controller (errno={getattr(ex, 'errno', ex)})")
        return
    
    if not ctrl.devices:
        print("No Govee devices discovered.")
        return
    
    dev = pick_device(ctrl, None)
    if not dev:
        print("Requested device not found.")
        return
    
    _cleanup_controller(ctrl)
    
    return dev


async def get_govee_device():
    '''For now just gets the first device, can be expanded to get a specific device by name or ID'''
    try:
        config = read_config(['govee_device_search_max_attempts'])
        max_attempts = config.get('govee_device_search_max_attempts', 3)

        for attempt in range(1, max_attempts + 1):
            if dev := await get_first_govee_device():
                # The `:=` operator is a Python 3.8+ feature that assigns and checks dev in a single line, reducing nesting and boilerplate
                return dev

            print(f"No Govee device found. Retrying - Attempt {attempt} of {max_attempts}")
            if attempt < max_attempts:
                await asyncio.sleep(1)
        else:
            # The `for...else` loop is a Pythonic construct that eliminates the need for a manual loop counter and conditional check!
            raise Exception("No Govee device found after multiple attempts.")
    
    except Exception as e:
        raise Exception(f"Error getting Govee device: {e}")



async def light_turn_on_handler():
    try:
        dev = await get_govee_device()
        await dev.turn_on()
        print("Turned ON")
        return {"success": True, "message": "Light turn-on command sent to Govee device."}
    except Exception as e:
        return {"success": False, "message": f"Error turning on Govee light: {e}"}


async def light_turn_off_handler():
    try:
        dev = await get_govee_device()
        await dev.turn_off()
        print("Turned OFF")
        return {"success": True, "message": "Light turn-off command sent to Govee device."}
    except Exception as e:
        return {"success": False, "message": f"Error turning off Govee light: {e}"}
    

async def light_set_brightness_handler(brightness: int):
    try:
        dev = await get_govee_device()
        await dev.set_brightness(brightness)
        print(f"Set brightness to {brightness}")
        return {"success": True, "message": f"Light brightness set to {brightness}."}
    except Exception as e:
        return {"success": False, "message": f"Error setting brightness: {e}"}

   
async def light_set_color_handler(red: int, green: int, blue: int):
    try:
        dev = await get_govee_device()
        await dev.set_color(red, green, blue)
        print(f"Set color to {red}, {green}, {blue}")
        return {"success": True, "message": f"Light color set to {red}, {green}, {blue}."}
    except Exception as e:
        return {"success": False, "message": f"Error setting color: {e}"}
    

async def light_set_temperature_handler(temperature: int):
    try:
        dev = await get_govee_device()
        await dev.set_temperature(temperature)
        print(f"Set temperature to {temperature}")
        return {"success": True, "message": f"Light temperature set to {temperature}."}
    except Exception as e:
        return {"success": False, "message": f"Error setting temperature: {e}"}


async def light_set_scene_handler(scene: str):
    try:
        dev = await get_govee_device()
        await dev.set_scene(scene)
        print(f"Set scene to {scene}")
        return {"success": True, "message": f"Light scene set to {scene}."}
    except Exception as e:
        return {"success": False, "message": f"Error setting scene: {e}"}


# async def main():
#     parser = argparse.ArgumentParser(description="Control a Govee light locally")
#     parser.add_argument("--action", choices=["on", "off"])
#     parser.add_argument("--id", help="Fingerprint or SKU (or substring) of device", default=None)
#     parser.add_argument("--bind", help="Local IPv4 to bind for discovery (NIC selection)", default=None)
#     parser.add_argument("--list", action="store_true", help="List discovered devices and exit")
#     args = parser.parse_args()

#     local_ip = args.bind or get_local_ip()
#     print(f"Using IP: {local_ip}")

#     ctrl = None
#     try:
#         ctrl = await discover_controller(local_ip)
#     except OSError as ex:
#         print(f"Failed to start controller (errno={getattr(ex, 'errno', ex)})")
#         return

#     if not ctrl.devices:
#         print("No Govee devices discovered.")
#         return

#     if args.list:
#         for d in ctrl.devices:
#             sku = getattr(d, "sku", "?")
#             fp = getattr(d, "fingerprint", "?")
#             host = getattr(d, "host", getattr(d, "ip", "?"))
#             print(f"- sku={sku} fp={fp} ip={host}")
#         return
    
#     if not args.action:
#         print("No --action provided. Use --list to see devices.")
#         return

#     dev = pick_device(ctrl, args.id)
#     if not dev:
#         print("Requested device not found.")
#         return
    
#     print(f"Controlling device: sku={getattr(dev, 'sku', '?')} fp={getattr(dev, 'fingerprint', '?')}")
#     if args.action == "on":
#         await dev.turn_on()
#         print("Turned ON")
#     else:
#         await dev.turn_off()
#         print("Turned OFF")

#     _cleanup_controller(ctrl)


# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     finally:
#         pass  # asyncio.run handles event loop shutdown