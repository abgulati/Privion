from web_app.services.smart_home_service import SmartHomeService

def verify_phase1():
    # Initialize Service by creating a new instance of SmartHomeService
    # This will be done both globally by the butler service and locally
    # by maintenance and testing services
    service = SmartHomeService()
    print("Smart Home Service Initialized.")

    # Query Existing Devices
    print("Querying existing devices...")
    devices = service.get_all_devices()
    for device in devices:
        print(f"Device: {device['name']} | Type: {device['type']} | Status: {device['is_online']} | Attributes: {device['attributes']}")

    # Query Existing Rooms
    print("Querying existing rooms...")
    rooms = service.get_all_rooms()
    for room in rooms:
        print(f"\tRoom: {room['name']}")

    # Query Existing Zones
    print("Querying existing zones...")
    zones = service.get_all_zones()
    for zone in zones:
        print(f"\tZone: {zone['name']}")

    # Query Existing Automations
    print("Querying existing automations...")
    automations = service.get_all_automations()
    for automation in automations:
        print(f"\tAutomation: {automation['name']}")
    
    print(f"\n--- Verification Results ---")
    print(f"Devices Found: {len(devices)}")
    print(f"Rooms Found: {len(rooms)}")
    print(f"Zones Found: {len(zones)}")
    print(f"Automations Found: {len(automations)}")
    
    if len(devices) > 0 and len(rooms) > 0 and len(zones) > 0 and len(automations) > 0:
        print("\nSUCCESS: Phase 1 Database & Service Operational.")
    else:
        print("\nFAIL: Data persistence failed.")

if __name__ == "__main__":
    verify_phase1()
