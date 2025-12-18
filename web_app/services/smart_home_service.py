import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

class SmartHomeService:
    """
    Stateless service class for managing the smart home database.
    Handles connection lifecycle and CRUD operations for Zones, Rooms, Devices, and Automations.
    """
    # Each house will have its own database    
    DB_NAME = "cameron_home.db"

    # Assuming the service is located in web_app/services/
    # DB will be in web_app/data/
    DB_PATH = Path(__file__).parent.parent / "data" / DB_NAME

    def __init__(self):
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a new database connection."""
        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """
        Initialize the database tables if they do not exist.
        This logic will need to evolve over time as the database schema changes.
        We need to think through how we will create databases during device setup.
        For now, this serves as a simple way to initialize the database on startup when the 
        global butler service creates a SmartHomeService instance.
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
