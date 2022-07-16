import socket
import random

ports=[]

def portConnection(port : int):
    s = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM)
            
    result = s.connect_ex(("localhost", port))
    if result == 0: return True
    return False

def determinePort():
    while True:
        randPort=random.randint(2000, 9000)
        if not portConnection(randPort):
            ports.append(randPort)
            return randPort