# scripts/launch_system.py
#!/usr/bin/env python3
"""
AI Query Handler System Launcher
Launch all components of the system
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

class SystemLauncher:
    def __init__(self):
        self.processes = {}
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}. Shutting down system...")
        self.stop_all()
        sys.exit(0)
    
    def check_prerequisites(self):
        """Check if system is ready to launch"""
        checks = {
            'Fine-tuned model': Path('models/fine_tuned').exists(),
            'Vector database': Path('data/vector_db').exists(), 
            'Configuration': Path('config/automation_config.json').exists(),
            'Dashboard': Path('dashboard/main_dashboard.py').exists(),
            'Scheduler': Path('src/email_scheduler.py').exists()
        }
        
        print("🔍 Prerequisites Check:")
        all_good = True
        
        for name, exists in checks.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {name}")
            if not exists:
                all_good = False
        
        return all_good
    
    def launch_dashboard(self):
        """Launch Streamlit dashboard"""
        print(" Launching dashboard...")
        
        try:
            cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/main_dashboard.py", 
                   "--server.port", "8501", "--server.address", "0.0.0.0"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            self.processes['dashboard'] = process
            print(" Dashboard launched on http://localhost:8501")
            return True
            
        except Exception as e:
            print(f" Failed to launch dashboard: {e}")
            return False
    
    def launch_scheduler(self):
        """Launch email scheduler"""
        print("⏰ Launching email scheduler...")
        
        try:
            cmd = [sys.executable, "src/email_scheduler.py"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            self.processes['scheduler'] = process
            print(" Email scheduler launched")
            return True
            
        except Exception as e:
            print(f" Failed to launch scheduler: {e}")
            return False
    
    def launch_all(self):
        """Launch all system components"""
        print(" LAUNCHING AI QUERY HANDLER SYSTEM")
        print("=" * 50)
        
        if not self.check_prerequisites():
            print(" Prerequisites check failed. Please complete setup first.")
            return False
        
        self.running = True
        
        # Launch components
        if not self.launch_dashboard():
            return False
        
        time.sleep(3)  # Wait for dashboard to start
        
        if not self.launch_scheduler():
            return False
        
        print("\n SYSTEM LAUNCHED SUCCESSFULLY!")
        print(" Dashboard: http://localhost:8501")
        print(" Email automation: Running in background")
        print(" Logs: Check logs/ directory")
        print("\n  Press Ctrl+C to stop all services")
        
        # Keep main thread alive and monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
            self.stop_all()
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            try:
                time.sleep(5)
                
                # Check process health
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        print(f" {name} process stopped unexpectedly")
                        
                        # Attempt restart
                        if name == 'dashboard':
                            print(f" Restarting {name}...")
                            self.launch_dashboard()
                        elif name == 'scheduler':
                            print(f" Restarting {name}...")
                            self.launch_scheduler()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f" Monitoring error: {e}")
    
    def stop_all(self):
        """Stop all running processes"""
        self.running = False
        
        print("🛑 Stopping all services...")
        
        for name, process in self.processes.items():
            try:
                print(f"  Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print(f"   {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"   Force killing {name}...")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                print(f"  ❌ Error stopping {name}: {e}")
        
        self.processes.clear()
        print(" All services stopped")
    
    def status(self):
        """Show system status"""
        print(" SYSTEM STATUS")
        print("=" * 30)
        
        if not self.processes:
            print(" No services running")
            return
        
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f" {name}: Running (PID: {process.pid})")
            else:
                print(f" {name}: Stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Query Handler System Launcher")
    parser.add_argument("--dashboard-only", action="store_true", 
                       help="Launch only the dashboard")
    parser.add_argument("--scheduler-only", action="store_true", 
                       help="Launch only the email scheduler")
    parser.add_argument("--status", action="store_true", 
                       help="Show system status")
    
    args = parser.parse_args()
    
    launcher = SystemLauncher()
    
    if args.status:
        launcher.status()
    elif args.dashboard_only:
        if launcher.check_prerequisites():
            launcher.launch_dashboard()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                launcher.stop_all()
    elif args.scheduler_only:
        if launcher.check_prerequisites():
            launcher.launch_scheduler()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                launcher.stop_all()
    else:
        launcher.launch_all()

if __name__ == "__main__":
    main()