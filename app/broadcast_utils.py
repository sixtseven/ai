import socket
import netifaces

def get_default_broadcast():
    gws = netifaces.gateways()
    default_iface =gws[netifaces.AF_INET][0][1]
    addrs = netifaces.ifaddresses(default_iface)[netifaces.AF_INET][0]
    return addrs['broadcast']

def send_broadcast(msg: bytes, port: int):
    broadcast_ip = get_default_broadcast()
    print(f"Broadcasting to {broadcast_ip}:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(msg, (broadcast_ip, port))
    sock.close()
