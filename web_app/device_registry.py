import web_app.ha_device_modules.kasa_plug as kasa_plug
import web_app.ha_device_modules.kasa_bulb as kasa_bulb

# Registry Helper
# Maps the vendor 'type' string from the database to the python module handling that device type.
# Each registered device module MUST implement a 'check_online_handler(ip_address)' function to properly check if the device is online.
# This removes any need for LLM  involvement in device status checks, creating a faster and more reliable system.

DEVICE_TYPE_HANDLERS = {
    "kasa_plug": kasa_plug,
    "kasa_bulb": kasa_bulb
}
