import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

class SmartHomeService:
    """
    Stateless service class for managing the smart home database.
    Handles connection lifecycle and CRUD operations for Zones, Rooms, Devices, and Automations.
    """
    # DB_NAME and DB_PATH are now instance variables determined at runtime
    # DB_NAME is the name of the database file - set to "smart_home.db"
    # DB_PATH is the path to the database file
    
    def __init__(self):
        self._configure_db_path()
        
        # If the DB does not exist, create it and initialize the tables
        if not self.DB_PATH.exists():
            print(f"[SmartHomeService] Database not found at {self.DB_PATH}. Creating...")
            self.init_db()
        else:
            print(f"[SmartHomeService] Database found at {self.DB_PATH}.")

    def _configure_db_path(self):
        """
        Reads storage_config.json to find the base_directory and sets DB_PATH.
        """
        try:
            # storage_config.json is up two parent directories
            config_path = Path(__file__).parent.parent / "storage_config.json"
            
            # Indicate error if the config path does not exist
            if not config_path.exists():
                raise FileNotFoundError(f"[SmartHomeService] CRITICAL: storage_config.json not found at {config_path}. Cannot determine database location.")

            with open(config_path, 'r') as f:
                config = json.load(f)
                base_dir = config.get('base_directory')
                
                if not base_dir:
                    raise ValueError("[SmartHomeService] CRITICAL: 'base_directory' key missing in storage_config.json.")
                    
                self.DB_PATH = Path(base_dir) / "smart_home.db"
            
        except Exception as e:
            # Re-raise exceptions to stop execution
            print(f"[SmartHomeService] FATAL ERROR configuring DB path: {e}")
            raise e

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a new database connection."""
        # Ensure parent directory exists for the new DB path
        if not self.DB_PATH.parent.exists():
             self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
             
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """
        Initialize the database tables.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Zones Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                )
            ''')
            
            # Rooms Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rooms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    zone_id INTEGER,
                    FOREIGN KEY (zone_id) REFERENCES zones (id)
                )
            ''')
            
            # Devices Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    type TEXT NOT NULL,
                    ip_address TEXT,
                    mac_address TEXT UNIQUE,
                    room_id INTEGER,
                    is_online BOOLEAN DEFAULT 0,
                    attributes TEXT,
                    FOREIGN KEY (room_id) REFERENCES rooms (id)
                )
            ''')
            
            # Automations Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS automations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    is_active BOOLEAN DEFAULT 1,
                    trigger_type TEXT,
                    config TEXT
                )
            ''')
            
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # ZONE Operations
    # ----------------------------------------------------------------
    def add_zone(self, name: str) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO zones (name) VALUES (?)", (name,))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM zones WHERE name = ?", (name,))
            return cursor.fetchone()['id']
        finally:
            conn.close()

    def get_all_zones(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM zones")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # ROOM Operations
    # ----------------------------------------------------------------
    def add_room(self, name: str, zone_id: Optional[int] = None) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO rooms (name, zone_id) VALUES (?, ?)", (name, zone_id))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM rooms WHERE name = ?", (name,))
            return cursor.fetchone()['id']
        finally:
            conn.close()

    def get_all_rooms(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rooms")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # DEVICE Operations
    # ----------------------------------------------------------------
    def add_device(self, name: str, device_type: str, ip_address: Optional[str] = None, 
                   mac_address: Optional[str] = None, room_id: Optional[int] = None, 
                   attributes: Optional[Dict] = None) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO devices (name, type, ip_address, mac_address, room_id, attributes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, device_type, ip_address, mac_address, room_id, json.dumps(attributes or {})))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Handle collision (Prefer Name update, fallback to MAC)
            cursor.execute("SELECT id FROM devices WHERE name = ?", (name,))
            row = cursor.fetchone()
            if not row and mac_address:
                cursor.execute("SELECT id FROM devices WHERE mac_address = ?", (mac_address,))
                row = cursor.fetchone()
            
            if row:
                device_id = row['id']
                cursor.execute("""
                    UPDATE devices 
                    SET type=?, ip_address=?, mac_address=?, room_id=?, attributes=?
                    WHERE id=?
                """, (device_type, ip_address, mac_address, room_id, json.dumps(attributes or {}), device_id))
                conn.commit()
                return device_id
            raise
        finally:
            conn.close()

    def get_device_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM devices WHERE name = ?", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_devices(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM devices")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def update_device_status(self, device_id: int, is_online: bool):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE devices SET is_online = ? WHERE id = ?", (is_online, device_id))
            conn.commit()
        finally:
            conn.close()

    def update_device_attributes(self, device_id: int, attributes: Dict):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE devices SET attributes = ? WHERE id = ?", (json.dumps(attributes), device_id))
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # AUTOMATION Operations
    # ----------------------------------------------------------------
    def add_automation(self, name: str, trigger_type: str, config: Dict, is_active: bool = True) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO automations (name, trigger_type, config, is_active)
                VALUES (?, ?, ?, ?)
            """, (name, trigger_type, json.dumps(config), is_active))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM automations WHERE name = ?", (name,))
            return cursor.fetchone()['id']
        finally:
            conn.close()

    def get_all_automations(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM automations")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
