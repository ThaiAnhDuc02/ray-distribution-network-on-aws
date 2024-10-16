import ray
import cv2
import numpy as np
import time
import psutil
import os
import socket
import subprocess  # Thêm để chạy lệnh hệ thống
from collections import defaultdict

@ray.remote
class ResourceMonitor:
    def __init__(self):
        self.node_ip = socket.gethostbyname(socket.gethostname())
        self.cpu_stats = []
        self.ram_stats = []
        self.swap_stats = []
        self.load_avg_stats = []
        self.start_time = time.time()

    def get_uptime(self):
        # Chạy lệnh uptime từ shell và trả về giá trị
        uptime_output = subprocess.check_output("uptime -p", shell=True)
        return uptime_output.decode().strip()

    def get_system_info(self):
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        load_avg = psutil.getloadavg()
        tasks = len(psutil.pids())
        
        ram_used = ram.used / (1024 * 1024 * 1024)  # Convert to GB
        swap_used = swap.used / (1024 * 1024 * 1024)  # Convert to GB
        
        self.cpu_stats.append(cpu_percent)
        self.ram_stats.append(ram_used)
        self.swap_stats.append(swap_used)
        self.load_avg_stats.append(load_avg)
        
        return {
            "node_ip": self.node_ip,
            "cpu_percent": cpu_percent,
            "ram_used": ram_used,
            "ram_total": ram.total / (1024 * 1024 * 1024),
            "swap_used": swap_used,
            "swap_total": swap.total / (1024 * 1024 * 1024),
            "tasks": tasks,
            "load_average": load_avg,
            "uptime": self.get_uptime()
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
            "swap_min": min(self.swap_stats),
            "swap_max": max(self.swap_stats),
            "swap_avg": sum(self.swap_stats) / len(self.swap_stats),
            "load_avg_min": [min(x) for x in zip(*self.load_avg_stats)],
            "load_avg_max": [max(x) for x in zip(*self.load_avg_stats)],
            "load_avg_avg": [sum(x)/len(x) for x in zip(*self.load_avg_stats)],
            "duration": end_time - self.start_time,
            "uptime": self.get_uptime()
        }

@ray.remote
def test(img):
    orb = cv2.AKAZE_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return des

def print_system_info(initial_info, final_stats):
    print(f"Node IP: {final_stats['node_ip']}")
    print(f"Uptime: {initial_info['uptime']}")
    print(f"Initial CPU Usage: {initial_info['cpu_percent']:.2f}%")
    print(f"Initial RAM Usage: {initial_info['ram_used']:.2f} GB / {initial_info['ram_total']:.2f} GB")
    print(f"Initial Swap Usage: {initial_info['swap_used']:.2f} GB / {initial_info['swap_total']:.2f} GB")
    print(f"Initial Tasks: {initial_info['tasks']}")
    print(f"Initial Load Average: {initial_info['load_average']}")
    print(f"CPU Usage - Min: {final_stats['cpu_min']:.2f}%, Max: {final_stats['cpu_max']:.2f}%, Avg: {final_stats['cpu_avg']:.2f}%")
    print(f"RAM Usage - Min: {final_stats['ram_min']:.2f} GB, Max: {final_stats['ram_max']:.2f} GB, Avg: {final_stats['ram_avg']:.2f} GB")
    print(f"Swap Usage - Min: {final_stats['swap_min']:.2f} GB, Max: {final_stats['swap_max']:.2f} GB, Avg: {final_stats['swap_avg']:.2f} GB")
    print(f"Load Average - Min: {final_stats['load_avg_min']}, Max: {final_stats['load_avg_max']}, Avg: {final_stats['load_avg_avg']}")
    print(f"Task Duration: {final_stats['duration']:.2f} seconds")

if __name__ == "__main__":
    ray.init(address="auto")  # This will connect to an existing Ray cluster

    # Get the number of CPUs and create ResourceMonitor actors
    num_cpus = int(ray.cluster_resources().get("CPU", 1))
    resource_monitors = [ResourceMonitor.remote() for _ in range(max(1, num_cpus))]

    # Ensure the image file exists
    img_path = "./tmp/test.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Get initial system info
    initial_infos = ray.get([monitor.get_system_info.remote() for monitor in resource_monitors])

    # Run tasks
    futures = [test.remote(img) for _ in range(100)]
    
    # Monitor resources while tasks are running
    while futures:
        done, futures = ray.wait(futures, timeout=1.0)
        ray.get([monitor.get_system_info.remote() for monitor in resource_monitors])

    # Get final statistics
    final_stats = ray.get([monitor.get_final_stats.remote() for monitor in resource_monitors])

    print("\nResource usage statistics:")
    for initial_info, final_stat in zip(initial_infos, final_stats):
        print_system_info(initial_info, final_stat)
        print("---")

    # Print Ray-specific information
    print("\nRay cluster resources:")
    print(ray.cluster_resources())

    ray.shutdown()
