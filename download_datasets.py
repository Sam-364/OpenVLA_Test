#!/usr/bin/env python3
"""
Dataset Downloader for OpenVLA Training
Downloads and prepares high-quality open-source robot manipulation datasets
"""

import os
import requests
import zipfile
import tarfile
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import hashlib

class DatasetDownloader:
    """Downloads and prepares datasets for OpenVLA training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'bridgedata_v2': {
                'name': 'BridgeData V2',
                'description': 'Large-scale robot manipulation dataset with 60K trajectories',
                'urls': {
                    'teleop': 'https://rail.eecs.berkeley.edu/datasets/bridge_release/data/teleop_data.zip',
                    'scripted': 'https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_data.zip'
                },
                'size_gb': 45,
                'format': 'zip',
                'structure': {
                    'images': '*/images*/im_*.jpg',
                    'actions': '*/actions.json',
                    'language': '*/language_annotations.json'
                }
            },
            'rlds': {
                'name': 'RLDS (Robot Learning Data Store)',
                'description': 'Standardized robot learning datasets',
                'urls': {
                    'main': 'https://github.com/google-research/rlds'
                },
                'size_gb': 20,
                'format': 'git',
                'structure': {
                    'examples': 'examples/*.py',
                    'datasets': 'datasets/*'
                }
            },
            'open_x_embodiment': {
                'name': 'Open X-Embodiment',
                'description': 'Multi-robot dataset with 1M+ trajectories',
                'urls': {
                    'paper': 'https://arxiv.org/abs/2310.08864',
                    'data': 'https://github.com/google-research/open_x_embodiment'
                },
                'size_gb': 100,
                'format': 'git',
                'structure': {
                    'datasets': 'datasets/*',
                    'models': 'models/*'
                }
            },
            'robosuite': {
                'name': 'RoboSuite',
                'description': 'Simulation environment with demonstration data',
                'urls': {
                    'main': 'https://github.com/ARISE-Initiative/robosuite'
                },
                'size_gb': 5,
                'format': 'git',
                'structure': {
                    'demos': 'robosuite/demos/*',
                    'models': 'robosuite/models/*'
                }
            }
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """Extract zip or tar archive"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            return True
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False
    
    def clone_git_repo(self, url: str, target_dir: Path) -> bool:
        """Clone a git repository"""
        try:
            subprocess.run([
                'git', 'clone', '--depth', '1', url, str(target_dir)
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error cloning {url}: {e}")
            return False
    
    def download_bridgedata_v2(self) -> bool:
        """Download BridgeData V2 dataset"""
        print("Downloading BridgeData V2...")
        
        dataset_dir = self.data_dir / "bridgedata_v2"
        dataset_dir.mkdir(exist_ok=True)
        
        success = True
        for data_type, url in self.datasets['bridgedata_v2']['urls'].items():
            print(f"Downloading {data_type} data...")
            
            # Download file
            filename = f"{data_type}_data.zip"
            filepath = dataset_dir / filename
            
            if not self.download_file(url, filepath):
                success = False
                continue
            
            # Extract
            extract_dir = dataset_dir / data_type
            if not self.extract_archive(filepath, extract_dir):
                success = False
                continue
            
            # Clean up zip file
            filepath.unlink()
            
            print(f"Downloaded and extracted {data_type} data")
        
        if success:
            print("BridgeData V2 download completed successfully!")
            self._create_dataset_info(dataset_dir, 'bridgedata_v2')
        
        return success
    
    def download_rlds(self) -> bool:
        """Download RLDS dataset"""
        print("Downloading RLDS...")
        
        dataset_dir = self.data_dir / "rlds"
        url = self.datasets['rlds']['urls']['main']
        
        if self.clone_git_repo(url, dataset_dir):
            print("RLDS download completed successfully!")
            self._create_dataset_info(dataset_dir, 'rlds')
            return True
        return False
    
    def download_open_x_embodiment(self) -> bool:
        """Download Open X-Embodiment dataset"""
        print("Downloading Open X-Embodiment...")
        
        dataset_dir = self.data_dir / "open_x_embodiment"
        url = self.datasets['open_x_embodiment']['urls']['data']
        
        if self.clone_git_repo(url, dataset_dir):
            print("Open X-Embodiment download completed successfully!")
            self._create_dataset_info(dataset_dir, 'open_x_embodiment')
            return True
        return False
    
    def download_robosuite(self) -> bool:
        """Download RoboSuite"""
        print("Downloading RoboSuite...")
        
        dataset_dir = self.data_dir / "robosuite"
        url = self.datasets['robosuite']['urls']['main']
        
        if self.clone_git_repo(url, dataset_dir):
            print("RoboSuite download completed successfully!")
            self._create_dataset_info(dataset_dir, 'robosuite')
            return True
        return False
    
    def _create_dataset_info(self, dataset_dir: Path, dataset_name: str):
        """Create dataset info file"""
        info = {
            'name': self.datasets[dataset_name]['name'],
            'description': self.datasets[dataset_name]['description'],
            'downloaded_at': str(Path().cwd()),
            'size_gb': self.datasets[dataset_name]['size_gb'],
            'structure': self.datasets[dataset_name]['structure']
        }
        
        info_file = dataset_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def list_datasets(self):
        """List available datasets"""
        print("Available datasets:")
        print("-" * 50)
        
        for key, dataset in self.datasets.items():
            print(f"Dataset: {dataset['name']}")
            print(f"Description: {dataset['description']}")
            print(f"Size: {dataset['size_gb']} GB")
            print(f"Format: {dataset['format']}")
            print("-" * 50)
    
    def check_downloaded_datasets(self):
        """Check which datasets are already downloaded"""
        print("Downloaded datasets:")
        print("-" * 50)
        
        for dataset_name in self.datasets.keys():
            dataset_dir = self.data_dir / dataset_name
            if dataset_dir.exists():
                info_file = dataset_dir / 'dataset_info.json'
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    print(f"✓ {info['name']} - {info['description']}")
                else:
                    print(f"✓ {dataset_name} (no info file)")
            else:
                print(f"✗ {dataset_name} - Not downloaded")
        
        print("-" * 50)
    
    def download_all(self):
        """Download all available datasets"""
        print("Downloading all datasets...")
        
        results = {
            'bridgedata_v2': self.download_bridgedata_v2(),
            'rlds': self.download_rlds(),
            'open_x_embodiment': self.download_open_x_embodiment(),
            'robosuite': self.download_robosuite()
        }
        
        print("\nDownload Summary:")
        print("-" * 30)
        for dataset, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{dataset}: {status}")
        
        return results
    
    def create_training_config(self, dataset_name: str):
        """Create training configuration for specific dataset"""
        config = {
            'dataset': {
                'name': dataset_name,
                'path': str(self.data_dir / dataset_name),
                'type': dataset_name
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 2e-5,
                'num_epochs': 27,
                'gradient_clip_val': 1.0,
                'checkpoint_dir': 'checkpoints',
                'log_dir': 'logs'
            },
            'model': {
                'hidden_size': 4096,
                'vision_hidden_size': 768,
                'action_bins': 256,
                'action_dim': 7
            }
        }
        
        config_file = self.data_dir / f"{dataset_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created training config: {config_file}")
        return config_file

def main():
    parser = argparse.ArgumentParser(description="Download datasets for OpenVLA training")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory to store datasets")
    parser.add_argument("--dataset", type=str, 
                       choices=['bridgedata_v2', 'rlds', 'open_x_embodiment', 'robosuite', 'all'],
                       help="Dataset to download")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--check", action="store_true",
                       help="Check downloaded datasets")
    parser.add_argument("--create_config", type=str,
                       help="Create training config for specific dataset")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_datasets()
    elif args.check:
        downloader.check_downloaded_datasets()
    elif args.create_config:
        downloader.create_training_config(args.create_config)
    elif args.dataset:
        if args.dataset == 'all':
            downloader.download_all()
        elif args.dataset == 'bridgedata_v2':
            downloader.download_bridgedata_v2()
        elif args.dataset == 'rlds':
            downloader.download_rlds()
        elif args.dataset == 'open_x_embodiment':
            downloader.download_open_x_embodiment()
        elif args.dataset == 'robosuite':
            downloader.download_robosuite()
    else:
        print("Please specify a dataset to download or use --list to see available options")
        print("Recommended: --dataset bridgedata_v2")

if __name__ == "__main__":
    main() 