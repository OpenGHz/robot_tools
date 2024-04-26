import os
import socketio
import socket
from typing import Union
import eventlet
from threading import Thread


class Socketer(object):

    @staticmethod
    def get_loop_ip():
        return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = 80
        all_ip = ("8.8.8.8", "114.114.114.114")
        ip_adr = None
        for ip in all_ip:
            try:
                s.connect((ip, port))
                ip_adr = s.getsockname()[0]
            except:
                ip_adr = None
            else:
                break
        return ip_adr

    @staticmethod
    def disable_proxy():
        proxy_names = (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
        for proxy_name in proxy_names:
            os.environ[proxy_name] = ""


class SimpleClient(object):

    def __init__(self, name="simple_client", host_port=None, disable_proxy=True):
        """
        name: str, client name
        disable_proxy: bool, disable proxy
        host_port: tuple, (host, port) where host is the server IP address in str type and port is the server port number in int type
        """
        if disable_proxy:
            Socketer.disable_proxy()

        self.name = name
        self.sio = socketio.Client()
        self.connected = False
        self.host_port = host_port

        @self.sio.event
        def connect():
            print(f"{self.name}: Connected to server")

        @self.sio.event
        def disconnect():
            print(f"{self.name}: Disconnected from server")

        @self.sio.event
        def response(data):
            print(f"{self.name}: Server response: {data}")

        if host_port is not None:
            self.connect(*host_port)

    def connect(self, host=None, port=7890):
        """e.g. ('192.168.112.195', 7890)"""
        if self.connected:
            print(f"{self.name}: Already connected to server:{self.host_port}.")
            return
        if host in ["auto", "AUTO"]:
            host = Socketer.get_ip()
            if host is None:
                raise Exception("Cannot auto get IP address.")
        elif host is None:
            host = "localhost"
        server_url = f"http://{host}:{port}"
        self.sio.connect(server_url)
        self.connected = True
        self.host_port = (host, port)

    def send_message(self, message: Union[str, bytes, list, dict, tuple]):
        self.sio.emit("message", {self.name: message})

    def wait(self):
        self.sio.wait()


class SimpleServer(object):
    def __init__(
        self, name="simple_server", host_port=None, disable_proxy=True, response=None
    ):
        """
        name: str, server name
        disable_proxy: bool, disable proxy
        host_port: tuple, (host, port) where host is the server IP address in str type and port is the server port number in int type
        response: message to send back to client, default is None which means no response
        """
        if disable_proxy:
            Socketer.disable_proxy()

        self.name = name
        self.sio = socketio.Server()
        self.app = socketio.WSGIApp(self.sio)

        @self.sio.event
        def connect(sid, environ):
            print(f"{self.name}: Client connected: {sid}")

        @self.sio.event
        def disconnect(sid):
            print(f"{self.name}: Client disconnected: {sid}")

        @self.sio.event
        def message(sid, data):
            print(f"{self.name}: Message from client {sid}: {data}")
            # "Received your message!"
            if response is not None:
                self.sio.emit(
                    "response", {f"{self.name}": response}, room=sid
                )

        if host_port is not None:
            self.start(*host_port)

    def start(self, host=None, port=7890, block=True):
        """e.g. ('192.168.112.195', 7890)"""
        if host in ["auto", "AUTO"]:
            host = Socketer.get_ip()
            if host is None:
                raise Exception("Cannot auto get IP address.")
        elif host is None:
            host = "localhost"
        self.server_url = f"http://{host}:{port}"
        self.host_port = (host, port)
        def start_server():
            eventlet.wsgi.server(eventlet.listen(self.host_port), self.app)
        if block:
            start_server()
        else:
            server_thread = Thread(target=start_server, daemon=True)
            server_thread.start()
            return server_thread


if __name__ == "__main__":

    server = SimpleServer(response="Received your message!")
    tr = server.start("AUTO", 7890, block=False)

    client = SimpleClient()
    client.connect("AUTO")
    client.send_message("Hello, server!")

    tr.join()
    client.wait()
