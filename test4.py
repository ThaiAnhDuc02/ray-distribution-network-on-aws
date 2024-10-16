import ray
import cv2
import numpy as np
import time
import psutil
import os
import socket
import subprocess  # Thêm module này để chạy lệnh hệ thống
from aptos_sdk.client import AptosClient
from aptos_sdk.account import Account
from aptos_sdk.transactions import TransactionPayloadEntryFunction

# Địa chỉ node Aptos (testnet hoặc mainnet)
APTOS_NODE_URL = "https://fullnode.testnet.aptoslabs.com/v1"
client = AptosClient(APTOS_NODE_URL)

# Địa chỉ private key của tài khoản Aptos (thay bằng private key của bạn)
PRIVATE_KEY = "0x437aec0c7d3c0e0f2c450216f3c5fe5dbba77dde37dfb0be3b3d53d5e7b4e835"
account = Account.load_key(PRIVATE_KEY)

# Địa chỉ module của contract và tên hàm cần gọi
MODULE_ADDRESS = "0xccfad7490745d7f76eb4b522d2904e19863a8246e8c84c17bcaa68d0dfa69fc9"
ENTRY_FUNCTION = "register_node"

@ray.remote
class ResourceMonitor:
    def __init__(self):
        self.node_ip = socket.gethostbyname(socket.gethostname())
        self.cpu_stats = []
        self.ram_stats = []
        self.uptime_stats = []  # Thêm danh sách này để lưu uptime
        self.start_time = time.time()

    def get_system_info(self):
        # Sử dụng lệnh hệ thống để lấy uptime
        uptime = subprocess.check_output("uptime -p", shell=True).decode().strip()
        
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024 * 1024 * 1024)  # Convert to GB
        
        self.cpu_stats.append(cpu_percent)
        self.ram_stats.append(ram_used)
        self.uptime_stats.append(uptime)

        return {
            "node_ip": self.node_ip,
            "cpu_percent": cpu_percent,
            "ram_used": ram_used,
            "uptime": uptime  # Trả về uptime
        }

    def get_final_stats(self):
        end_time = time.time()
        return {
            "node_ip": self.node_ip,
            "cpu_min": min(self.cpu_stats),
            "cpu_max": max(self.cpu_stats),
            "cpu_avg": sum(self.cpu_stats) / len(self.cpu_stats),
            "ram_min": min(self.ram_stats),
            "ram_max": max(self.ram_stats),
            "ram_avg": sum(self.ram_stats) / len(self.ram_stats),
            "uptime_start": self.uptime_stats[0],  # Lấy uptime khi bắt đầu
            "uptime_end": self.uptime_stats[-1],   # Lấy uptime khi kết thúc
            "duration": end_time - self.start_time
        }

@ray.remote
def test(img):
    orb = cv2.AKAZE_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return des

def print_system_info(initial_info, final_stats):
    print(f"Node IP: {final_stats['node_ip']}")
    print(f"Initial CPU Usage: {initial_info['cpu_percent']:.2f}%")
    print(f"Initial RAM Usage: {initial_info['ram_used']:.2f} GB")
    print(f"Uptime Start: {final_stats['uptime_start']}")
    print(f"Uptime End: {final_stats['uptime_end']}")
    print(f"CPU Usage - Min: {final_stats['cpu_min']:.2f}%, Max: {final_stats['cpu_max']:.2f}%, Avg: {final_stats['cpu_avg']:.2f}%")
    print(f"RAM Usage - Min: {final_stats['ram_min']:.2f} GB, Max: {final_stats['ram_max']:.2f} GB, Avg: {final_stats['ram_avg']:.2f} GB")
    print(f"Task Duration: {final_stats['duration']:.2f} seconds")

def register_node(info):
    # Chuẩn bị các đối số để gửi vào contract
    arguments = [
        info["node_ip"],                  # IP của node
        str(info["cpu_percent"]),         # CPU usage %
        str(info["ram_used"])             # RAM usage (GB)
    ]
    
    # Tạo payload cho transaction
    payload = TransactionPayloadEntryFunction(
        MODULE_ADDRESS,
        ENTRY_FUNCTION,
        arguments
    )

    # Ký và gửi transaction lên blockchain
    txn_hash = client.submit_transaction(account, payload)
    
    # Chờ transaction được xác nhận
    client.wait_for_transaction(txn_hash)
    
    print(f"Node registered successfully with transaction hash: {txn_hash}")

if __name__ == "__main__":
    ray.init(address="auto")  # Kết nối tới Ray cluster

    # Lấy số lượng CPUs và tạo ResourceMonitor actors
    num_cpus = int(ray.cluster_resources().get("CPU", 1))
    resource_monitors = [ResourceMonitor.remote() for _ in range(max(1, num_cpus))]

    # Đảm bảo rằng file ảnh tồn tại
    img_path = "./tmp/test.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Lấy thông tin hệ thống ban đầu và đăng ký node
    initial_infos = ray.get([monitor.get_system_info.remote() for monitor in resource_monitors])

    # Đăng ký từng node với thông tin tài nguyên đã thu thập
    for initial_info in initial_infos:
        register_node(initial_info)

    # Chạy các tác vụ
    futures = [test.remote(img) for _ in range(100)]
    
    # Theo dõi tài nguyên khi các tác vụ đang chạy
    while futures:
        done, futures = ray.wait(futures, timeout=1.0)
        ray.get([monitor.get_system_info.remote() for monitor in resource_monitors])

    # Lấy các thống kê cuối cùng
    final_stats = ray.get([monitor.get_final_stats.remote() for monitor in resource_monitors])

    print("\nResource usage statistics:")
    for initial_info, final_stat in zip(initial_infos, final_stats):
        print_system_info(initial_info, final_stat)
        print("---")

    # In ra thông tin tài nguyên của Ray cluster
    print("\nRay cluster resources:")
    print(ray.cluster_resources())

    ray.shutdown()
