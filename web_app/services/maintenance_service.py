import sys
import os
import asyncio
import time
import signal
import threading

# Ensure parent directory (web_app) is in sys.path so we can import 'services', 'device_registry', etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from services.smart_home_service import SmartHomeService
from device_registry import DEVICE_TYPE_HANDLERS
import global_states


SHUTDOWN_EVENT = threading.Event()

def shutdown_signal_handler(signum, frame):
    print(f"\n[Maintenance] Received signal {signum}. Shutting down gracefully...")
    SHUTDOWN_EVENT.set()

async def check_all_devices(service: SmartHomeService):
    devices = service.get_all_devices()
    tasks = []
    
    print(f"\n[Maintenance] Checking {len(devices)} devices...")
    
    for device in devices:
        dev_type = device['type']
        ip_addr = device.get('ip_address')
        
        # Look up handler in registry
        handler_module = DEVICE_TYPE_HANDLERS.get(dev_type)
        
        if handler_module and ip_addr:
            # Create a task for this device check
            # We wrap it to track the device ID for the update
            print(f"  Checking {device['name']} (Type: {dev_type}) with IP: {ip_addr}")
            tasks.append(check_and_update(service, device['id'], device['name'], handler_module, ip_addr, device.get('mac_address')))
        else:
            print(f"  Skipping {device['name']} (Type: {dev_type}): No handler or missing IP.")

    if tasks:
        await asyncio.gather(*tasks)

async def check_and_update(service, device_id, name, module, ip, mac_address):
    try:
        # Call the module's handler
        result = await module.check_online_handler(ip)
        is_online = result.get('success', False) and result.get('message') == 'online'
        
        # SELF-HEALING: If offline and we have a MAC address, try to find it
        if not is_online and mac_address and hasattr(module, 'find_device_ip'):
            print(f"  {name} is offline at {ip}. Attempting self-healing via MAC {mac_address}...")
            new_ip = await module.find_device_ip(mac_address)
            
            if new_ip and new_ip != ip:
                print(f"  [SELF-HEALING] Found {name} at new IP: {new_ip}! Updating DB...")
                # Update DB with new IP
                conn = service._get_connection()
                try:
                    conn.execute("UPDATE devices SET ip_address = ? WHERE id = ?", (new_ip, device_id))
                    conn.commit()
                finally:
                    conn.close()
                
                # Update status to Online
                is_online = True
            else:
                print(f"  [SELF-HEALING] Could not find {name} via discovery.")

        print(f"  {name}: {'Online' if is_online else 'Offline'}")
        
        # Update DB
        service.update_device_status(device_id, is_online)
        
        # Update attributes if available
        if is_online and result.get('state'):
             service.update_device_attributes(device_id, result['state'])
        
    except Exception as e:
        print(f"  Error checking {name}: {e}")

def maintenance_loop(once=False):
    print("--- Starting Maintenance Service (Registry Pattern) ---")
    service = SmartHomeService()
    
    # Windows-specific fix for "Event loop is closed" RuntimeError
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    while not SHUTDOWN_EVENT.is_set():
        try:
            if global_states.get_system_busy():
                print("[Maintenance] System is busy. Skipping maintenance cycle.")
                # Wait 3 seconds, or until shutdown event is set
                SHUTDOWN_EVENT.wait(3)
                continue

            # Run the async check loop
            asyncio.run(check_all_devices(service))
            print("[Maintenance] Cycle Complete. Waiting 10 seconds...")

        except Exception as e:
            print(f"[Maintenance] Critical Error in loop: {e}")
        
        if once:
            break
            
        # Wait 10 seconds, or until shutdown event is set
        SHUTDOWN_EVENT.wait(10)
    
    print("[Maintenance] Service Shutdown Complete.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_signal_handler)
    signal.signal(signal.SIGTERM, shutdown_signal_handler)
    maintenance_loop()
