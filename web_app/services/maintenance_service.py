from services.smart_home_service import SmartHomeService
from device_registry import DEVICE_TYPE_HANDLERS
import asyncio
import time
import threading
import sys
import global_state

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

def start_background_service():
    """Starts the maintenance loop in a background daemon thread."""
    try:
        print("Initializing Maintenance Service Background Thread...")
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        thread.start()
        print("Maintenance Service Background Thread Started.")
        return thread
    except Exception as e:
        print(f"Failed to start Maintenance Service Background Thread: {e}")
        return None

def maintenance_loop(once=False):
    print("--- Starting Maintenance Service (Registry Pattern) ---")
    service = SmartHomeService()
    
    # Windows-specific fix for "Event loop is closed" RuntimeError
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    while True:
        try:
            if global_state.get_system_busy():
                print("[Maintenance] System is busy. Skipping maintenance cycle.")
                time.sleep(60)
                continue

            # Run the async check loop
            asyncio.run(check_all_devices(service))
            print("[Maintenance] Cycle Complete. Waiting 60 seconds...")

        except Exception as e:
            print(f"[Maintenance] Critical Error in loop: {e}")
        
        if once:
            break
            
        time.sleep(60)

if __name__ == "__main__":
    # arg to allow running once from CLI (useful for testing)
    run_once = "--once" in sys.argv
    maintenance_loop(once=run_once)
