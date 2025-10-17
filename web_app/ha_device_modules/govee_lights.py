import sys
import asyncio
import logging
import socket
import argparse
from contextlib import suppress

from govee_local_api import GoveeController

# # Windows: reuse_port is not supported; make asyncio ignore it.
# if sys.platform == "win32":
#     try:
#         import asyncio.base_events as _base_events
#         _base_events._set_reuseport = lambda sock: None  # no-op on Windows
#     except Exception:
#         pass

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
                logging.getLogger("govee_control").warning(
                    "Ignoring WinError 10022 during multicast setup; continuing without multicast join."
                )
                # Discovery uses unicast replies to listening socket; missing the multicast join only affects passive announcements, not active discovery.
                return
            raise

    GoveeController.connection_made = patched

_patch_govee_on_windows()


BROADCAST_ADDR = "239.255.255.250"
TARGET_PORT = 4001
LISTENING_PORT = 4002
DISCOVERY_TIMEOUT = 5

log = logging.getLogger("govee_control")
logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    ctrl = GoveeController(
        loop=loop,
        logger=log,
        listening_address=local_ip,
        broadcast_address=BROADCAST_ADDR,
        broadcast_port=TARGET_PORT,
        listening_port=LISTENING_PORT,
        discovery_enabled=True,
        discovery_interval=1,
        update_enabled=False,
    )
    await ctrl.start()
    # Wait up to DISCOVERY_TIMEOUT seconds for at least one device
    for _ in range(DISCOVERY_TIMEOUT * 2):
        if ctrl.devices:
            break
        await asyncio.sleep(0.5)
    return ctrl


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


async def main():
    parser = argparse.ArgumentParser(description="Control a Govee light locally")
    parser.add_argument("--action", choices=["on", "off"])
    parser.add_argument("--id", help="Fingerprint or SKU (or substring) of device", default=None)
    parser.add_argument("--bind", help="Local IPv4 to bind for discovery (NIC selection)", default=None)
    parser.add_argument("--list", action="store_true", help="List discovered devices and exit")
    args = parser.parse_args()

    local_ip = args.bind or get_local_ip()
    log.info(f"Using IP: {local_ip}")

    ctrl = None
    try:
        ctrl = await discover_controller(local_ip)
    except OSError as ex:
        log.error(f"Failed to start controller (errno={getattr(ex, 'errno', ex)})")
        return

    if not ctrl.devices:
        log.error("No Govee devices discovered.")
        return

    if args.list:
        for d in ctrl.devices:
            sku = getattr(d, "sku", "?")
            fp = getattr(d, "fingerprint", "?")
            host = getattr(d, "host", getattr(d, "ip", "?"))
            log.info(f"- sku={sku} fp={fp} ip={host}")
        return
    
    if not args.action:
        log.error("No --action provided. Use --list to see devices.")
        return

    dev = pick_device(ctrl, args.id)
    if not dev:
        log.error("Requested device not found.")
        return
    
    log.info(f"Controlling device: sku={getattr(dev, 'sku', '?')} fp={getattr(dev, 'fingerprint', '?')}")
    if args.action == "on":
        await dev.turn_on()
        log.info("Turned ON")
    else:
        await dev.turn_off()
        log.info("Turned OFF")

    _cleanup_controller(ctrl)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        pass  # asyncio.run handles event loop shutdown