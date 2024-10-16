import ray
import cv2
import numpy as np
import time
import psutil
import os
import socket

@ray.remote
class ResourceMonitor:
    def get_system_info(self):
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used = ram.used / (1024 * 1024 * 1024)  # Convert to GB
        return {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "ram_used": ram_used,
            "node_ip": socket.gethostbyname(socket.gethostname())
        }

@ray.remote
def test(img):
    orb = cv2.AKAZE_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return des

def print_system_info(info):
    print(f"Node IP: {info['node_ip']}")
    print(f"CPU Usage: {info['cpu_percent']}%")
    print(f"RAM Usage: {info['ram_percent']}%")
    print(f"RAM Used: {info['ram_used']:.2f} GB")

if __name__ == "__main__":
    ray.init(address="auto")  # This will connect to an existing Ray cluster

    start_time = time.time()

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

    futures = [test.remote(img) for _ in range(100)]

    while futures:
        done, futures = ray.wait(futures, timeout=1.0)
        
        # Collect system info from all nodes
        system_infos = ray.get([monitor.get_system_info.remote() for monitor in resource_monitors])
        
        print("\nCurrent system info across all nodes:")
        for info in system_infos:
            print_system_info(info)
            print("---")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nExecution time: {execution_time:.2f} seconds")

    print("\nFinal system info across all nodes:")
    system_infos = ray.get([monitor.get_system_info.remote() for monitor in resource_monitors])
    for info in system_infos:
        print_system_info(info)
        print("---")

    # Print Ray-specific information
    print("\nRay cluster resources:")
    print(ray.cluster_resources())

    # Get the current process on the head node
    process = psutil.Process(os.getpid())
    
    # Get memory info for the head node
    memory_info = process.memory_info()
    
    print(f"\nMemory used by this process on head node: {memory_info.rss / (1024 * 1024):.2f} MB")
    print(f"CPU time used by this process on head node: {sum(process.cpu_times()[:2]):.2f} seconds")

    ray.shutdown()
