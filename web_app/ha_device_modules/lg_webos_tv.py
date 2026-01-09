import json
import time
import struct
import socket

from pathlib import Path
from typing import Optional
from contextlib import closing  # used to close the socket automatically
from aiowebostv.webos_client import WebOsClient

###############---------Security Stuff---------########################
KEYS_FILE = Path("ha_device_modules/butler_webos_keys.json")

def _load_client_key(host: str) -> str | None:
    if not KEYS_FILE.exists():
        return None
    try:
        data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        return data.get(host)
    except Exception:
        return None

def _save_client_key(host: str, client_key: str) -> None:
    data = {}
    if KEYS_FILE.exists():
        try:
            data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[host] = client_key
    KEYS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


#####################---------Network Discovery---------###########################
SSDPMCAST_ADDR = ('239.255.255.250', 1900)  # Default SSDP multicast address and port

def get_default_local_ip() -> str:
    """
    Creates a socket and connects to Google's public DNS server (8.8.8.8) on port 80 to get the default local IP address.

    socket.AF_INET is the address family for IPv4.
    socket.SOCK_DGRAM is the socket type for UDP.
    Args:
        None
    Returns:
        The default local IP address of the machine.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def _parse_ssdp_headers(raw: str) -> dict[str, str]:
    """
    Parse the headers from an SSDP response.
    Args:
        raw: The raw SSDP response.
    Returns:
        A dictionary of the headers.
    """
    headers: dict[str, str] = {}
    for line in raw.split("\r\n")[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return headers


def _looks_like_lg_webos(headers: dict[str, str]) -> bool:
    """
    Check if the headers look like a LG WebOS TV.
    Args:
        headers: The headers to check.
    Returns:
        True if the headers look like a LG WebOS TV, False otherwise.
    """
    server = headers.get("server", "")
    st = headers.get("st", "")    # Service Type
    usn = headers.get("usn", "")    # Service Unique Number
    href = headers.get("location", "")   # HTTP Location Header
    # Heuristics commonly seen on LG webOS TVs
    text = f"{server} {st} {usn} {href}".lower()
    return (
        "urn:lge-com:service:webos-second-screen:1" in text
        or "webos" in text
        or "lge" in text
        or ("lg" in text and "tv" in text)
    )


def _ssdp_msearch(local_ip: str, sts: str, mx: int = 2, timeout: float = 3.0, repeats: int = 3) -> list[tuple[str, dict[str, str]]]:
    """
    Send a simple SSDP M-SEARCH for the given ST.
    Args:
        local_ip: The local IP address of the machine.
        sts: The Service Types to search for.
        mx: The maximum number of responses to wait for.
        timeout: The timeout in seconds.
        repeats: The number of times to send the SSDP M-SEARCH.
    Returns:
        A list of tuples containing the IP address and headers of the devices found, list of (responder_ip, headers).
    """
    msg_tpl = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDPMCAST_ADDR[0]}:{SSDPMCAST_ADDR[1]}\r\n"
        'MAN: "ssdp:discover"\r\n'
        f"MX: {mx}\r\n"
        "ST: {st}\r\n"
        "\r\n"
    )

    results: list[tuple[str, dict[str, str]]] = []
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)) as sock:
        sock.bind((local_ip, 0))
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack('b', 2))

        for st in sts:
            pkt = msg_tpl.format(st=st).encode("ascii")
            for _ in range(repeats):
                sock.sendto(pkt, SSDPMCAST_ADDR)
                time.sleep(0.15)

        end = time.time() + timeout
        while time.time() < end:
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                break
            headers = _parse_ssdp_headers(data.decode(errors="ignore"))
            results.append((addr[0], headers))
        return results


def discover_lg_webos_tv_ips(local_ip: Optional[str] = None, timeout: float = 3.0) -> list[str]:
    """
    Discover the IP addresses of LG WebOS TVs on the network.
    Args:
        timeout: The timeout in seconds.
    Returns:
        A list of the IP addresses of the LG WebOS TVs.
    """

    # Try LG-specific service types first, then fallback to ssdp:all
    local_ip = local_ip or get_default_local_ip()
    sts = [
        "urn:lge-com:service:webos-second-screen:1",
        "urn:schemas-upnp-org:device:MediaRenderer:1",
        "upnp:rootdevice",
        "ssdp:all",
    ]
    hits = _ssdp_msearch(local_ip, sts, timeout=timeout, repeats=3)
    ips: list[str] = []
    seen: set[str] = set()
    for ip, headers in hits:
        if _looks_like_lg_webos(headers) and ip not in seen:
            seen.add(ip)
            ips.append(ip)
    return ips


def discover_webos_ip(local_ip: Optional[str] = None, timeout: float = 6.0) -> Optional[str]:
    """
    Discover the IP address of a LG WebOS TV on the network.
    Args:
        local_ip: The local IP address of the machine.
        timeout: The timeout in seconds.
    Returns:
        The IP address of the LG WebOS TV.
    """
    ips = discover_lg_webos_tv_ips(local_ip=local_ip, timeout=timeout)
    return ips[0] if ips else None
    

#####################---------WebOS Client---------###########################
async def webos_pair_connect_and_power_off_async(host: str) -> dict:
    """
    Pair and connect to a LG WebOS TV and power it off.
    Args:
        host: The IP address of the LG WebOS TV.
    Returns:
        A dictionary with a success flag and a message.
    """
    client_key = _load_client_key(host)
    client = WebOsClient(host, client_key=client_key)
    try:
        connected = await client.connect()
        if not connected:
            return {"success": False, "message": "Failed to connect to TV. Make sure to accept the pairing prompt on the TV!"}

        # If this was first-time pairing, accept the prompt on the TV.
        # After acceptance, the client_key will be set, and is saved to the keys file.
        if not client_key and client.client_key:
            _save_client_key(host, client.client_key)

        await client.power_off()
        return {"success": True, "message": "Power-off command sent to LG WebOS TV."}
    except Exception as e:
        return {"success": False, "message": f"Error sending power-off command to LG WebOS TV: {e}"}
    finally:
        await client.disconnect()