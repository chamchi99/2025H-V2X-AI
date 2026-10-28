import socket
import json
from time import sleep

address = ("106.248.49.98",50020)
address = ("192.168.254.2",40001)
address = ("192.168.254.2",40001)
# address = ("223.171.51.242",  40001)
class SendSocket:
    SendAddress = ("127.0.0.1", 50020)
    UDPSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def addHdr(cls, msg, type):
        byteMsg = bytes([0xEF]) + bytes([0x01]) + bytes([type]) + len(msg).to_bytes(4, 'big') + str.encode(msg)
        return byteMsg
    
    @classmethod
    def addHdrBytes(cls, msg, type):
        byteMsg = bytes([0xEF]) + bytes([0x01]) + bytes([type]) + len(msg).to_bytes(4, 'big') + msg
        return byteMsg
    
    @classmethod
    def sendto(cls, msgType, msg, address = SendAddress):
        msgBytes = cls.addHdr(msg, msgType)
        cls.UDPSocket.sendto(msgBytes, address)

    @classmethod
    def sendFileData(cls, msgType, fileName, address = SendAddress):
        file = open(fileName, 'r')

             
        msg = file.read()

        print(msg)
        if fileName == "cctv-traffic.json":
            msg = msg
            print(msg)
        if fileName == "cctv-traffic-vds.json":
            msg =  msg[1:-1]
            print(msg)

        msgBytes = cls.addHdr(msg, msgType)
        # print(msgBytes)
        # print("")
        # print("")
        # print("")
        # print("")
        # print("")
        # print("")
        # print("")
        cls.UDPSocket.sendto(msgBytes, address)

# json 데이터 스트링으로 직접 입력 및 전송
def test1():
    sock = SendSocket()
    while True:
        sock.sendto(0x02, "{\"traffic\":[{\"GPS\":[126.611507,37.528178],\"bbox\":[0,0,0,0],\"cctv\":1.0,\"class\":1.0,\"id\":500,\"lane\":1.0,\"speed\":777.0,\"timestamp\":1698039859},{\"GPS\":[126.611507,37.528278],\"bbox\":[0,0,0,0],\"cctv\":1.0,\"class\":2.0,\"id\":501.000,\"lane\":1.0,\"speed\":777.0,\"timestamp\":1698039859}]}", address)
        # sock.sendto(0x01, "{\"lane1_+speed\": 30.0, \"lane1_count\": 1, \"lane2_speed\": -999, \"lane2_count\": 0, \"lane3_speed\": 33.0, \"lane3_count\": 1, \"merge_count\": 0, \"timestamp\": 1698225881}, {\"lane1_speed\": 30.0, \"lane1_count\": 1, \"lane2_speed\": -999, \"lane2_count\": 0, \"lane3_speed\": 33.0, \"lane3_count\": 1, \"merge_count\": 0, \"timestamp\": 1698225881}", address)
        sleep(0.1)

# json 파일 데이터 읽어 전송
cctv_traffic_json = "cctv_traffic.json"
cctv_traffic_json = "cctv-traffic.json"
# cctv_traffic_json = "cctv_traffic_sample.json"
# cctv_traffic_json = "test.json"
cctv_traffic_vds_json = "cctv_traffic_vds.json"


def test2():
    sock = SendSocket()
    while True:
        sock.sendFileData(0x02, cctv_traffic_json, address)
        # sock.sendFileData(0x02, cctv_traffic_vds_json, address)
        sleep(0.1)
        

if __name__ == "__main__":
    test2()
    
    