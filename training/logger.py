import sys
import os
from pathlib import Path
from datetime import datetime
import yaml

class TrainingLogger:
    
    def __init__(self, log_dir: str, config_path: str = None, config_dict: dict = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"training_log_{timestamp}.txt"
        
        self.file = open(self.log_file, 'w', encoding='utf-8')
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        self.terminal = sys.stdout
        
        self._write_header()
        
        if config_path:
            self._write_config_from_path(config_path)
        elif config_dict:
            self._write_config_from_dict(config_dict)
        
        print(f"\n{'='*80}")
        print(f" Training log will be saved to: {self.log_file}")
        print(f"{'='*80}\n")
    
    def _write_header(self):
        header = f"""
{'='*80}
DEEPFAKE DETECTION - TRAINING LOG
{'='*80}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log File: {self.log_file}
{'='*80}

"""
        self.file.write(header)
        self.file.flush()
    
    def _write_config_from_path(self, config_path: str):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            self.file.write(f"\n{'='*80}\n")
            self.file.write(f"CONFIGURATION (from {config_path})\n")
            self.file.write(f"{'='*80}\n\n")
            self.file.write(config_content)
            self.file.write(f"\n{'='*80}\n")
            self.file.write(f"END OF CONFIGURATION\n")
            self.file.write(f"{'='*80}\n\n")
            self.file.flush()
            
        except Exception as e:
            self.file.write(f"ERROR: Could not read config file: {e}\n\n")
            self.file.flush()
    
    def _write_config_from_dict(self, config_dict: dict):
        try:
            self.file.write(f"\n{'='*80}\n")
            self.file.write(f"CONFIGURATION\n")
            self.file.write(f"{'='*80}\n\n")
            self.file.write(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
            self.file.write(f"\n{'='*80}\n")
            self.file.write(f"END OF CONFIGURATION\n")
            self.file.write(f"{'='*80}\n\n")
            self.file.flush()
            
        except Exception as e:
            self.file.write(f"ERROR: Could not write config: {e}\n\n")
            self.file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        
        if '\r' not in message and not message.startswith('Epoch') or 'Epoch' in message and '%|' not in message:
            self.file.write(message)
            self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        footer = f"""
{'='*80}
Training Completed
End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        self.file.write(footer)
        self.file.flush()
        
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        self.file.close()
        
        print(f"\n{'='*80}")
        print(f" Training log saved to: {self.log_file}")
        print(f"{'='*80}\n")
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            import traceback
            error_msg = f"\n{'='*80}\n"
            error_msg += f"ERROR OCCURRED DURING TRAINING\n"
            error_msg += f"{'='*80}\n"
            error_msg += f"Exception Type: {exc_type.__name__}\n"
            error_msg += f"Exception Message: {exc_val}\n\n"
            error_msg += "Traceback:\n"
            error_msg += ''.join(traceback.format_tb(exc_tb))
            error_msg += f"{'='*80}\n"
            
            self.file.write(error_msg)
            self.terminal.write(error_msg)
        
        self.close()
        return False
