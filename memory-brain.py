# memory_system.py - 智能记忆系统（带压缩存储和智能增强）- 修复版
import json
import pickle
import gzip
import zipfile
import lz4.frame
import hashlib
import base64
import msgpack
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from collections import defaultdict, deque
import heapq
import re
import math
import traceback

# 尝试导入 zstandard，如果失败则提供一个替代方案
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️ zstandard 库未安装，zstd 压缩算法将不可用")

class CompressedMemory:
    """压缩记忆存储（增强版）- 修复版"""
    
    COMPRESSION_ALGORITHMS = {
        'gzip': {
            'compress': lambda data: gzip.compress(data, compresslevel=6),
            'decompress': gzip.decompress,
            'extension': '.gz'
        },
        'lz4': {
            'compress': lambda data: lz4.frame.compress(data),
            'decompress': lz4.frame.decompress,
            'extension': '.lz4'
        },
        'none': {
            'compress': lambda data: data,
            'decompress': lambda data: data,
            'extension': ''
        }
    }
    
    def __init__(self, config):
        self.config = config
        self.memory_dir = config.get_path("memory_dir")
        self.compression_config = config.get("compression", {})
        self.algorithm = self.compression_config.get("algorithm", "gzip")
        
        # 添加 zstd 算法（如果可用）
        if ZSTD_AVAILABLE:
            self.COMPRESSION_ALGORITHMS['zstd'] = {
                'compress': lambda data: zstd.compress(data),
                'decompress': zstd.decompress,
                'extension': '.zst'
            }
        
        if self.algorithm not in self.COMPRESSION_ALGORITHMS:
            self.algorithm = "gzip"
        
        # 内存缓存（增强）
        self.memory_cache = {}
        self.cache_size_limit = 2000
        self.access_counter = defaultdict(int)
        self.access_timestamps = {}
        
        # 统计信息
        self.stats = {
            "total_compressed": 0,
            "total_decompressed": 0,
            "compression_ratio": 1.0,
            "saved_space_mb": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_access_time": 0,
            "failed_decompressions": 0,
            "corrupted_files": set()  # 记录损坏的文件
        }
        
        # 创建记忆目录
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化时检查并修复损坏的文件
        self.check_and_repair_files()
    
    def get_optimal_algorithm(self, data_size: int, importance: float) -> str:
        """根据数据大小和重要性选择最佳压缩算法"""
        if importance > 0.8:  # 高重要性数据
            return 'gzip'  # gzip更可靠
        elif ZSTD_AVAILABLE and data_size > 1024 * 1024:  # 大于1MB
            return 'zstd'  # zstd压缩比高
        elif data_size < 1024:  # 小于1KB
            return 'lz4'  # lz4速度快
        else:
            return self.algorithm
    
    def compress_data(self, data: Any, importance: float = 0.5) -> Tuple[bytes, str]:
        """压缩数据（智能选择算法）"""
        try:
            # 序列化数据
            serialized = msgpack.packb(data, use_bin_type=True)
            
            # 根据数据大小和重要性选择算法
            data_size = len(serialized)
            optimal_algo = self.get_optimal_algorithm(data_size, importance)
            
            # 检查算法是否可用
            if optimal_algo not in self.COMPRESSION_ALGORITHMS:
                optimal_algo = self.algorithm
            
            # 应用压缩
            if optimal_algo != "none":
                compressed = self.COMPRESSION_ALGORITHMS[optimal_algo]['compress'](serialized)
                
                # 更新统计
                original_size = len(serialized)
                compressed_size = len(compressed)
                ratio = compressed_size / original_size if original_size > 0 else 1.0
                
                self.stats["total_compressed"] += 1
                self.stats["compression_ratio"] = 0.9 * self.stats["compression_ratio"] + 0.1 * ratio
                self.stats["saved_space_mb"] += (original_size - compressed_size) / (1024 * 1024)
                
                return compressed, optimal_algo
            else:
                return serialized, "none"
                
        except Exception as e:
            print(f"压缩失败: {e}")
            return pickle.dumps(data), "pickle"
    
    def decompress_data(self, compressed_data: bytes, algorithm: str = None) -> Any:
        """解压数据（修复版）"""
        start_time = datetime.now()
        
        try:
            # 如果指定了算法且可用
            if algorithm and algorithm in self.COMPRESSION_ALGORITHMS:
                try:
                    decompressed = self.COMPRESSION_ALGORITHMS[algorithm]['decompress'](compressed_data)
                    data = msgpack.unpackb(decompressed, raw=False)
                    
                    # 计算访问时间
                    access_time = (datetime.now() - start_time).total_seconds() * 1000
                    self.stats["avg_access_time"] = 0.9 * self.stats["avg_access_time"] + 0.1 * access_time
                    self.stats["total_decompressed"] += 1
                    
                    return data
                    
                except Exception as e:
                    print(f"⚠️ 使用 {algorithm} 解压失败，尝试自动检测: {e}")
            
            # 自动检测算法
            algorithms_to_try = []
            
            # 如果指定了算法但不是第一个，先尝试指定的
            if algorithm and algorithm != "none":
                algorithms_to_try.append(algorithm)
            
            # 添加所有可用算法
            for algo in ['gzip', 'lz4', 'zstd', 'none']:
                if algo in self.COMPRESSION_ALGORITHMS and algo != algorithm:
                    algorithms_to_try.append(algo)
            
            # 尝试 pickle 作为最后手段
            algorithms_to_try.append('pickle')
            
            # 尝试每个算法
            for algo in algorithms_to_try:
                try:
                    if algo == "none":
                        data = msgpack.unpackb(compressed_data, raw=False)
                    elif algo == "pickle":
                        data = pickle.loads(compressed_data)
                    else:
                        decompressed = self.COMPRESSION_ALGORITHMS[algo]['decompress'](compressed_data)
                        data = msgpack.unpackb(decompressed, raw=False)
                    
                    # 计算访问时间
                    access_time = (datetime.now() - start_time).total_seconds() * 1000
                    self.stats["avg_access_time"] = 0.9 * self.stats["avg_access_time"] + 0.1 * access_time
                    self.stats["total_decompressed"] += 1
                    
                    print(f"✅ 使用 {algo} 算法成功解压数据")
                    return data
                    
                except Exception as e:
                    continue
            
            # 所有算法都失败
            self.stats["failed_decompressions"] += 1
            print(f"❌ 所有解压方法都失败")
            return None
            
        except Exception as e:
            self.stats["failed_decompressions"] += 1
            print(f"解压失败: {e}")
            return None
    
    def check_and_repair_files(self):
        """检查并修复损坏的文件"""
        print("🔄 检查记忆文件完整性...")
        
        repaired = 0
        corrupted = 0
        
        for filepath in self.memory_dir.glob("*"):
            if filepath.suffix in ['.meta', '.json']:
                continue
            
            # 检查文件大小
            if filepath.stat().st_size == 0:
                print(f"⚠️ 发现空文件: {filepath.name}")
                corrupted += 1
                self.stats["corrupted_files"].add(str(filepath))
                continue
            
            # 尝试读取文件
            try:
                with open(filepath, 'rb') as f:
                    compressed = f.read()
                
                # 查找对应的元数据文件
                meta_file = filepath.with_suffix('.meta')
                algorithm = self.algorithm
                
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_info = json.load(f)
                            algorithm = meta_info.get("algorithm", self.algorithm)
                    except:
                        algorithm = self.algorithm
                
                # 尝试解压
                data = self.decompress_data(compressed, algorithm)
                
                if data is None:
                    print(f"⚠️ 文件可能已损坏: {filepath.name}")
                    corrupted += 1
                    self.stats["corrupted_files"].add(str(filepath))
                else:
                    repaired += 1
                    
            except Exception as e:
                print(f"❌ 检查文件 {filepath.name} 时出错: {e}")
                corrupted += 1
                self.stats["corrupted_files"].add(str(filepath))
        
        print(f"✅ 文件检查完成: {repaired} 个正常, {corrupted} 个可能损坏")
        return repaired, corrupted
    
    def save_memory(self, memory_id: str, data: Any, metadata: Dict = None) -> bool:
        """保存记忆（增强版本）"""
        try:
            # 准备数据
            memory_data = {
                "data": data,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "version": "1.2",  # 版本更新
                "access_count": 0,
                "last_accessed": None
            }
            
            # 获取重要性
            importance = metadata.get("importance", 0.5) if metadata else 0.5
            
            # 压缩（智能选择算法）
            compressed, algorithm_used = self.compress_data(memory_data, importance)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = self.COMPRESSION_ALGORITHMS.get(algorithm_used, {}).get('extension', '.bin')
            filename = f"{memory_id}_{timestamp}_{importance:.2f}{file_ext}"
            filepath = self.memory_dir / filename
            
            # 保存文件（包含算法信息）
            with open(filepath, 'wb') as f:
                f.write(compressed)
            
            # 保存算法信息到元文件
            meta_info = {
                "algorithm": algorithm_used,
                "size": len(compressed),
                "importance": importance,
                "memory_id": memory_id,
                "original_size": len(pickle.dumps(memory_data)),
                "compression_ratio": len(compressed) / max(1, len(pickle.dumps(memory_data))),
                "created_at": datetime.now().isoformat()
            }
            meta_file = filepath.with_suffix('.meta')
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, ensure_ascii=False, indent=2)
            
            # 更新缓存
            self.memory_cache[memory_id] = {
                "data": data,
                "metadata": metadata,
                "filepath": filepath,
                "accessed": datetime.now(),
                "access_count": 0,
                "importance": importance,
                "algorithm": algorithm_used,
                "compressed_size": len(compressed)
            }
            
            # 智能缓存管理
            self.manage_cache()
            
            print(f"✅ 保存记忆: {memory_id} ({len(compressed)} 字节)")
            return True
            
        except Exception as e:
            print(f"❌ 保存记忆失败: {e}")
            traceback.print_exc()
            return False
    
    def load_memory(self, memory_id: str, pattern: str = None) -> Optional[Any]:
        """加载记忆（带缓存优化）"""
        # 检查缓存
        if memory_id in self.memory_cache:
            cache_entry = self.memory_cache[memory_id]
            cache_entry["accessed"] = datetime.now()
            cache_entry["access_count"] += 1
            self.access_counter[memory_id] += 1
            self.stats["cache_hits"] += 1
            return cache_entry["data"]
        
        self.stats["cache_misses"] += 1
        
        # 查找文件
        files = []
        if pattern:
            files = list(self.memory_dir.glob(f"{pattern}*"))
        else:
            files = list(self.memory_dir.glob(f"{memory_id}_*"))
        
        if not files:
            print(f"❌ 未找到记忆文件: {memory_id}")
            return None
        
        # 按时间排序（最新的优先），同时考虑重要性
        def file_score(filepath):
            # 从文件名提取重要性
            try:
                importance_match = re.search(r'_(\d+\.\d{2})', filepath.name)
                importance = float(importance_match.group(1)) if importance_match else 0.5
            except:
                importance = 0.5
            
            # 检查是否在损坏文件列表中
            if str(filepath) in self.stats["corrupted_files"]:
                return -1000  # 损坏文件得分很低
            
            mtime = filepath.stat().st_mtime
            # 重要性高的文件优先，时间新的优先
            return importance * 1000 + mtime
        
        files.sort(key=file_score, reverse=True)
        
        for filepath in files:
            try:
                # 检查是否在损坏文件列表中
                if str(filepath) in self.stats["corrupted_files"]:
                    print(f"⚠️ 跳过已知损坏文件: {filepath.name}")
                    continue
                
                # 读取算法信息
                meta_file = filepath.with_suffix('.meta')
                algorithm = self.algorithm
                importance = 0.5
                
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_info = json.load(f)
                            algorithm = meta_info.get("algorithm", self.algorithm)
                            importance = meta_info.get("importance", 0.5)
                    except:
                        algorithm = self.algorithm
                        importance = 0.5
                
                # 读取文件
                with open(filepath, 'rb') as f:
                    compressed = f.read()
                
                # 解压
                memory_data = self.decompress_data(compressed, algorithm)
                
                if memory_data:
                    # 更新访问统计
                    memory_data["access_count"] = memory_data.get("access_count", 0) + 1
                    memory_data["last_accessed"] = datetime.now().isoformat()
                    
                    # 更新缓存
                    self.memory_cache[memory_id] = {
                        "data": memory_data["data"],
                        "metadata": memory_data.get("metadata", {}),
                        "filepath": filepath,
                        "accessed": datetime.now(),
                        "access_count": memory_data["access_count"],
                        "importance": importance,
                        "algorithm": algorithm,
                        "compressed_size": len(compressed)
                    }
                    
                    self.access_counter[memory_id] = memory_data["access_count"]
                    
                    # 异步更新文件中的访问统计
                    self.update_access_stats_async(filepath, memory_data["access_count"])
                    
                    print(f"✅ 加载记忆: {memory_id} ({len(compressed)} 字节)")
                    return memory_data["data"]
                else:
                    print(f"⚠️ 解压失败，标记为损坏: {filepath.name}")
                    self.stats["corrupted_files"].add(str(filepath))
            except Exception as e:
                print(f"❌ 加载记忆失败 {filepath.name}: {e}")
                self.stats["corrupted_files"].add(str(filepath))
                continue
        
        return None
    
    def update_access_stats_async(self, filepath: Path, access_count: int):
        """异步更新访问统计"""
        try:
            meta_file = filepath.with_suffix('.meta')
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_info = json.load(f)
                
                meta_info["access_count"] = access_count
                meta_info["last_accessed"] = datetime.now().isoformat()
                
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta_info, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索记忆（智能搜索）"""
        results = []
        query_lower = query.lower()
        
        # 1. 首先搜索缓存
        for memory_id, cache_entry in self.memory_cache.items():
            try:
                data_str = str(cache_entry["data"]).lower()
                metadata_str = str(cache_entry.get("metadata", {})).lower()
                
                # 计算相关性分数
                score = self.calculate_relevance_score(query_lower, data_str, metadata_str, cache_entry)
                if score > 0:
                    results.append({
                        "file": cache_entry.get("filepath", Path()).name,
                        "data": cache_entry["data"],
                        "metadata": cache_entry.get("metadata", {}),
                        "created": cache_entry.get("metadata", {}).get("created_at", ""),
                        "relevance_score": score,
                        "source": "cache"
                    })
            except:
                continue
        
        # 2. 搜索文件系统（如果缓存结果不够）
        if len(results) < limit:
            for filepath in self.memory_dir.glob("*"):
                if filepath.suffix in ['.meta', '.json']:
                    continue
                    
                # 跳过损坏文件
                if str(filepath) in self.stats["corrupted_files"]:
                    continue
                
                try:
                    # 检查是否已在缓存结果中
                    if any(r["file"] == filepath.name for r in results):
                        continue
                        
                    # 读取算法信息
                    algorithm = self.algorithm
                    meta_file = filepath.with_suffix('.meta')
                    if meta_file.exists():
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_info = json.load(f)
                            algorithm = meta_info.get("algorithm", self.algorithm)
                    
                    # 读取文件
                    with open(filepath, 'rb') as f:
                        compressed = f.read()
                    
                    # 解压并搜索
                    memory_data = self.decompress_data(compressed, algorithm)
                    if memory_data:
                        data_str = str(memory_data).lower()
                        metadata = memory_data.get("metadata", {})
                        metadata_str = str(metadata).lower()
                        
                        # 计算相关性分数
                        cache_entry = {
                            "importance": meta_info.get("importance", 0.5) if meta_file.exists() else 0.5,
                            "access_count": meta_info.get("access_count", 0) if meta_file.exists() else 0
                        }
                        score = self.calculate_relevance_score(query_lower, data_str, metadata_str, cache_entry)
                        
                        if score > 0:
                            results.append({
                                "file": filepath.name,
                                "data": memory_data["data"],
                                "metadata": metadata,
                                "created": metadata.get("created_at", ""),
                                "relevance_score": score,
                                "source": "file"
                            })
                            
                            if len(results) >= limit * 2:  # 多收集一些用于排序
                                break
                except:
                    continue
        
        # 3. 按相关性排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:limit]
    
    def calculate_relevance_score(self, query: str, data_str: str, metadata_str: str, cache_entry: Dict) -> float:
        """计算相关性分数"""
        score = 0.0
        
        # 1. 直接匹配
        if query in data_str:
            score += 2.0
        if query in metadata_str:
            score += 1.5
        
        # 2. 部分匹配
        words = query.split()
        for word in words:
            if len(word) > 2:
                if word in data_str:
                    score += 0.5
                if word in metadata_str:
                    score += 0.3
        
        # 3. 重要性权重
        importance = cache_entry.get("importance", 0.5)
        score *= (0.5 + importance)  # 重要性高的记忆得分更高
        
        # 4. 访问频率权重
        access_count = cache_entry.get("access_count", 0)
        if access_count > 0:
            score *= (1.0 + math.log(1 + access_count) / 10)
        
        return score
    
    def cleanup_old_memories(self, days_old: int = 30, keep_important: bool = True):
        """清理旧记忆（智能清理）"""
        cutoff = datetime.now() - timedelta(days=days_old)
        
        deleted = 0
        kept = 0
        
        for filepath in self.memory_dir.glob("*"):
            if filepath.suffix in ['.meta', '.json']:
                continue
                
            try:
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                
                # 检查是否重要
                is_important = False
                meta_file = filepath.with_suffix('.meta')
                if meta_file.exists() and keep_important:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_info = json.load(f)
                        if meta_info.get("importance", 0) > 0.8:  # 高重要性记忆
                            is_important = True
                
                if mtime < cutoff and not is_important:
                    filepath.unlink()
                    
                    # 删除元文件
                    if meta_file.exists():
                        meta_file.unlink()
                    
                    deleted += 1
                    
                    # 从缓存中移除
                    for key in list(self.memory_cache.keys()):
                        if self.memory_cache[key].get("filepath") == filepath:
                            del self.memory_cache[key]
                            
                    # 从损坏文件列表中移除
                    if str(filepath) in self.stats["corrupted_files"]:
                        self.stats["corrupted_files"].remove(str(filepath))
                else:
                    kept += 1
            except:
                continue
        
        print(f"记忆清理: 删除了 {deleted} 个旧记忆，保留了 {kept} 个记忆")
        return deleted
    
    def manage_cache(self):
        """智能缓存管理"""
        if len(self.memory_cache) <= self.cache_size_limit:
            return
        
        # 按访问频率和重要性计算分数，淘汰低分项目
        cache_scores = []
        for key, entry in self.memory_cache.items():
            # 分数 = 访问频率 * 0.4 + 重要性 * 0.4 + 时间衰减 * 0.2
            access_count = self.access_counter.get(key, 0)
            importance = entry.get("importance", 0.5)
            
            # 时间衰减（最近访问的分数高）
            time_since_access = (datetime.now() - entry.get("accessed", datetime.now())).total_seconds()
            time_score = max(0, 1 - time_since_access / (24 * 3600))  # 24小时衰减
            
            score = access_count * 0.4 + importance * 0.4 + time_score * 0.2
            cache_scores.append((score, key))
        
        # 排序，保留高分项目
        cache_scores.sort(reverse=True)
        keep_keys = {key for _, key in cache_scores[:self.cache_size_limit]}
        
        # 移除低分项目
        for key in list(self.memory_cache.keys()):
            if key not in keep_keys:
                del self.memory_cache[key]
    
    def consolidate_important_memories(self):
        """巩固重要记忆"""
        important_memories = []
        
        # 收集高重要性记忆
        for filepath in self.memory_dir.glob("*"):
            if filepath.suffix in ['.meta', '.json']:
                continue
                
            # 跳过损坏文件
            if str(filepath) in self.stats["corrupted_files"]:
                continue
                
            try:
                meta_file = filepath.with_suffix('.meta')
                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_info = json.load(f)
                        
                    if meta_info.get("importance", 0) > 0.7:  # 中等以上重要性
                        important_memories.append((filepath, meta_info))
            except:
                continue
        
        # 重新压缩重要记忆，使用更可靠的算法
        consolidated = 0
        for filepath, meta_info in important_memories:
            try:
                # 读取原数据
                with open(filepath, 'rb') as f:
                    compressed = f.read()
                
                algorithm = meta_info.get("algorithm", self.algorithm)
                memory_data = self.decompress_data(compressed, algorithm)
                
                if memory_data:
                    # 重新压缩为gzip（更可靠）
                    importance = meta_info.get("importance", 0.5)
                    recompressed, new_algorithm = self.compress_data(memory_data, importance)
                    
                    # 如果新算法不同，保存新文件
                    if new_algorithm != algorithm:
                        new_filename = filepath.stem.rsplit('_', 1)[0] + f"_{importance:.2f}{self.COMPRESSION_ALGORITHMS[new_algorithm]['extension']}"
                        new_filepath = filepath.parent / new_filename
                        
                        with open(new_filepath, 'wb') as f:
                            f.write(recompressed)
                        
                        # 更新元数据
                        meta_info["algorithm"] = new_algorithm
                        meta_info["consolidated_at"] = datetime.now().isoformat()
                        meta_info["original_algorithm"] = algorithm
                        
                        with open(new_filepath.with_suffix('.meta'), 'w', encoding='utf-8') as f:
                            json.dump(meta_info, f, ensure_ascii=False, indent=2)
                        
                        # 删除旧文件
                        filepath.unlink()
                        if filepath.with_suffix('.meta').exists():
                            filepath.with_suffix('.meta').unlink()
                        
                        # 更新缓存中的文件路径
                        for key, cache_entry in self.memory_cache.items():
                            if cache_entry.get("filepath") == filepath:
                                cache_entry["filepath"] = new_filepath
                                cache_entry["algorithm"] = new_algorithm
                        
                        consolidated += 1
            except Exception as e:
                print(f"巩固记忆失败 {filepath}: {e}")
        
        print(f"巩固了 {consolidated} 个重要记忆")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_files = len([f for f in self.memory_dir.glob("*") if f.suffix not in ['.meta', '.json']])
        
        cache_hit_rate = 0
        total_accesses = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_accesses > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_accesses
        
        avg_access_time = self.stats.get("avg_access_time", 0)
        if isinstance(avg_access_time, float):
            avg_access_time_ms = f"{avg_access_time:.2f}ms"
        else:
            avg_access_time_ms = f"{avg_access_time}ms"
        
        return {
            **self.stats,
            "cache_size": len(self.memory_cache),
            "total_files": total_files,
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "avg_access_time": avg_access_time_ms,
            "algorithm": self.algorithm,
            "corrupted_files_count": len(self.stats["corrupted_files"]),
            "access_counter_size": len(self.access_counter),
            "zstd_available": ZSTD_AVAILABLE
        }

class IntelligentMemory:
    """智能记忆管理系统（增强版）"""
    
    def __init__(self, config):
        self.config = config
        self.compressed_memory = CompressedMemory(config)
        
        # 记忆分类（增强）
        self.memory_categories = {
            "conversation": {"name": "对话记忆", "importance": 0.6},
            "knowledge": {"name": "知识记忆", "importance": 0.8},
            "preference": {"name": "偏好记忆", "importance": 0.7},
            "learning": {"name": "学习记忆", "importance": 0.9},
            "system": {"name": "系统记忆", "importance": 0.5},
            "concept": {"name": "概念记忆", "importance": 0.85},
            "fact": {"name": "事实记忆", "importance": 0.75},
            "experience": {"name": "经验记忆", "importance": 0.7}
        }
        
        # 短期记忆（最近对话）- 增强
        self.short_term_memory = deque(maxlen=100)
        self.short_term_weights = {}
        
        # 记忆索引（增强）
        self.memory_index = defaultdict(list)
        self.concept_network = defaultdict(set)
        self.semantic_links = defaultdict(list)
        
        # 记忆巩固队列
        self.consolidation_queue = deque(maxlen=50)
        
        # 初始化
        self.load_concept_network()
        print(f"✅ 记忆系统初始化完成")
    
    def load_concept_network(self):
        """加载概念网络"""
        network_file = self.config.get_path("memory_dir") / "concept_network.json"
        if network_file.exists():
            try:
                with open(network_file, 'r', encoding='utf-8') as f:
                    network_data = json.load(f)
                    self.concept_network = defaultdict(set, {k: set(v) for k, v in network_data.get("concept_network", {}).items()})
                    self.semantic_links = defaultdict(list, network_data.get("semantic_links", {}))
                print(f"📚 加载概念网络: {len(self.concept_network)} 个概念")
            except Exception as e:
                print(f"⚠️ 加载概念网络失败: {e}")
    
    def save_concept_network(self):
        """保存概念网络"""
        network_file = self.config.get_path("memory_dir") / "concept_network.json"
        try:
            network_data = {
                "concept_network": {k: list(v) for k, v in self.concept_network.items()},
                "semantic_links": dict(self.semantic_links),
                "updated_at": datetime.now().isoformat()
            }
            with open(network_file, 'w', encoding='utf-8') as f:
                json.dump(network_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存概念网络失败: {e}")
    
    def remember(self, category: str, key: str, data: Any, 
                 metadata: Dict = None, tags: List[str] = None) -> bool:
        """记住信息（增强版）"""
        # 准备元数据
        full_metadata = {
            "category": category,
            "key": key,
            "tags": tags or [],
            "importance": metadata.get("importance", 
                self.memory_categories.get(category, {}).get("importance", 0.5)) if metadata else 
                self.memory_categories.get(category, {}).get("importance", 0.5),
            "context": metadata.get("context", "") if metadata else "",
            "related_concepts": [],
            **({} if metadata is None else metadata)
        }
        
        # 提取概念并建立网络
        if category == "knowledge" or category == "concept":
            concepts = self.extract_concepts(data, metadata)
            if concepts:
                full_metadata["concepts"] = concepts
                self.update_concept_network(key, concepts, data)
        
        # 建立语义链接
        if tags:
            self.create_semantic_links(key, tags, category, data)
        
        # 保存到压缩存储
        memory_id = f"{category}_{key}"
        success = self.compressed_memory.save_memory(memory_id, data, full_metadata)
        
        if success:
            # 更新索引
            self.memory_index[category].append(memory_id)
            
            # 添加标签索引
            if tags:
                for tag in tags:
                    self.memory_index[f"tag_{tag}"].append(memory_id)
            
            # 如果是对话，添加到短期记忆
            if category == "conversation":
                self.short_term_memory.append({
                    "key": key,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                    "importance": full_metadata["importance"]
                })
                # 设置短期记忆权重
                self.short_term_weights[key] = 1.0
            
            # 添加到巩固队列
            if full_metadata["importance"] > 0.7:
                self.consolidation_queue.append(memory_id)
        
        return success
    
    def extract_concepts(self, data: Any, metadata: Dict) -> List[str]:
        """从数据中提取概念"""
        concepts = []
        
        # 从数据中提取
        if isinstance(data, dict):
            text_parts = []
            for value in data.values():
                if isinstance(value, str):
                    text_parts.append(value)
            text = " ".join(text_parts)
        elif isinstance(data, str):
            text = data
        else:
            text = str(data)
        
        # 提取中文概念（2-5个字）
        chinese_concepts = re.findall(r"[\u4e00-\u9fa5]{2,5}", text)
        concepts.extend(chinese_concepts[:5])
        
        # 从标签中提取
        if metadata and "tags" in metadata:
            for tag in metadata["tags"]:
                if isinstance(tag, str) and len(tag) >= 2:
                    concepts.append(tag)
        
        # 去重
        return list(set(concepts))
    
    def update_concept_network(self, key: str, concepts: List[str], data: Any):
        """更新概念网络"""
        for concept in concepts:
            # 添加概念到网络
            self.concept_network[concept].add(key)
            
            # 建立概念之间的关联
            for other_concept in concepts:
                if concept != other_concept:
                    self.concept_network[concept].add(other_concept)
        
        # 定期保存网络
        if len(self.concept_network) % 10 == 0:
            self.save_concept_network()
    
    def create_semantic_links(self, key: str, tags: List[str], category: str, data: Any):
        """创建语义链接"""
        # 查找有相同标签的记忆
        for tag in tags:
            tag_key = f"tag_{tag}"
            if tag_key in self.memory_index:
                related_memories = self.memory_index[tag_key][-5:]
                if related_memories:
                    self.semantic_links[key].extend(related_memories)
    
    def recall(self, category: str = None, key: str = None, 
               tag: str = None, limit: int = 5, context: str = None, 
               sort_by: str = None, **kwargs) -> List[Any]:
        """回忆信息（增强版）"""
        memories = []
        
        if key and category:
            # 精确回忆
            memory_id = f"{category}_{key}"
            data = self.compressed_memory.load_memory(memory_id)
            if data:
                memories.append(data)
        
        elif tag:
            # 按标签回忆（考虑上下文）
            tag_key = f"tag_{tag}"
            if tag_key in self.memory_index:
                memory_ids = self.memory_index[tag_key]
                
                # 如果有上下文，进行排序
                if context:
                    scored_memories = []
                    for memory_id in memory_ids:
                        data = self.compressed_memory.load_memory(memory_id)
                        if data:
                            # 计算与上下文的相似度
                            if isinstance(data, dict):
                                data_str = str(data)
                            else:
                                data_str = str(data)
                            
                            similarity = self.calculate_similarity(context, data_str)
                            scored_memories.append((similarity, data))
                    
                    # 按相似度排序
                    scored_memories.sort(reverse=True)
                    memories = [data for _, data in scored_memories[:limit]]
                else:
                    for memory_id in memory_ids[:limit]:
                        data = self.compressed_memory.load_memory(memory_id)
                        if data:
                            memories.append(data)
        
        elif category:
            # 按分类回忆（考虑重要性）
            if category in self.memory_index:
                memory_ids = self.memory_index[category]
                
                # 加载并评分
                scored_memories = []
                for memory_id in memory_ids:
                    data = self.compressed_memory.load_memory(memory_id)
                    if data:
                        # 从缓存获取重要性
                        cache_entry = self.compressed_memory.memory_cache.get(memory_id.split('_', 1)[1])
                        importance = cache_entry.get("importance", 0.5) if cache_entry else 0.5
                        scored_memories.append((importance, data))
                
                # 按重要性排序
                scored_memories.sort(reverse=True)
                memories = [data for _, data in scored_memories[:limit]]
        
        else:
            # 获取短期记忆（加权）
            short_term = list(self.short_term_memory)
            if short_term:
                # 根据排序方式处理
                if sort_by == "recent":
                    short_term.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                elif sort_by == "important":
                    short_term.sort(key=lambda x: x.get("importance", 0.5), reverse=True)
                
                # 按时间和权重排序
                weighted_memories = []
                for memory in short_term:
                    key_val = memory.get("key", "")
                    weight = self.short_term_weights.get(key_val, 1.0)
                    timestamp = memory.get("timestamp", "")
                    
                    # 时间衰减
                    if timestamp:
                        try:
                            mem_time = datetime.fromisoformat(timestamp)
                            time_diff = (datetime.now() - mem_time).total_seconds()
                            time_factor = max(0, 1 - time_diff / (3600 * 24))
                        except:
                            time_factor = 0.5
                    else:
                        time_factor = 0.5
                    
                    score = weight * time_factor
                    weighted_memories.append((score, memory.get("data")))
                
                if not sort_by:
                    weighted_memories.sort(reverse=True)
                
                memories = [data for _, data in weighted_memories[:limit]]
        
        return memories
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        if not text1 or not text2:
            return 0.0
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # 计算共同词汇
        words1 = set(re.findall(r'\w+', text1_lower))
        words2 = set(re.findall(r'\w+', text2_lower))
        
        if not words1 or not words2:
            return 0.0
        
        common_words = words1.intersection(words2)
        
        # Jaccard相似度
        similarity = len(common_words) / len(words1.union(words2))
        
        return similarity
    
    def search(self, query: str, category: str = None, limit: int = 10) -> List[Dict]:
        """搜索记忆（增强版）"""
        # 使用压缩内存的智能搜索
        results = self.compressed_memory.search_memories(query, limit * 2)
        
        if category:
            results = [r for r in results 
                      if r["metadata"].get("category") == category]
        
        # 提取概念，查找相关记忆
        if len(results) < limit:
            concepts = self.extract_concepts(query, {})
            for concept in concepts[:3]:
                if concept in self.concept_network:
                    related_keys = list(self.concept_network[concept])[:5]
                    for key in related_keys:
                        if '_' in key:
                            parts = key.split('_', 1)
                            if len(parts) == 2:
                                cat, mem_key = parts
                                data = self.compressed_memory.load_memory(key)
                                if data:
                                    # 检查是否已存在
                                    if not any(r["metadata"].get("key") == mem_key for r in results):
                                        results.append({
                                            "file": f"related_{key}",
                                            "data": data,
                                            "metadata": {"category": cat, "key": mem_key},
                                            "created": datetime.now().isoformat(),
                                            "relevance_score": 0.3,
                                            "source": "concept_network"
                                        })
        
        return results[:limit]
    
    def learn_from_conversation(self, user_input: str, ai_response: str, 
                               context: Dict = None):
        """从对话中学习（增强版）"""
        # 生成记忆键
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        memory_key = f"conv_{hashlib.md5(user_input.encode()).hexdigest()[:8]}_{timestamp}"
        
        # 提取关键词和概念
        keywords = self.extract_keywords(user_input + " " + ai_response)
        concepts = self.extract_concepts(user_input + " " + ai_response, {})
        
        # 分析对话类型
        conversation_type = self.analyze_conversation_type(user_input, ai_response)
        
        # 保存对话
        conversation_data = {
            "user": user_input,
            "ai": ai_response,
            "context": context or {},
            "keywords": keywords,
            "concepts": concepts,
            "type": conversation_type,
            "length": len(user_input) + len(ai_response)
        }
        
        # 根据对话类型设置重要性
        importance = 0.6
        if conversation_type == "knowledge_sharing":
            importance = 0.8
        elif conversation_type == "teaching":
            importance = 0.9
        elif conversation_type == "question_answer":
            importance = 0.7
        
        self.remember(
            category="conversation",
            key=memory_key,
            data=conversation_data,
            metadata={
                "importance": importance,
                "context": "对话学习",
                "emotion": self.detect_emotion(user_input),
                "conversation_type": conversation_type
            },
            tags=keywords + ["conversation"]
        )
        
        # 尝试提取知识
        self.extract_knowledge(user_input, ai_response, keywords, concepts, conversation_type)
        
        # 更新短期记忆权重
        self.update_short_term_weights(memory_key, importance)
    
    def analyze_conversation_type(self, user_input: str, ai_response: str) -> str:
        """分析对话类型"""
        user_lower = user_input.lower()
        ai_lower = ai_response.lower()
        
        teaching_patterns = ["是", "就是", "指的是", "意味着", "定义为"]
        if any(pattern in user_lower for pattern in teaching_patterns):
            return "teaching"
        
        knowledge_indicators = ["知识", "信息", "数据", "事实", "原理"]
        if any(indicator in user_lower or indicator in ai_lower for indicator in knowledge_indicators):
            return "knowledge_sharing"
        
        question_indicators = ["吗", "什么", "怎么", "为什么", "如何", "是不是"]
        if any(indicator in user_lower for indicator in question_indicators):
            return "question_answer"
        
        chat_indicators = ["你好", "再见", "谢谢", "哈哈", "嗯"]
        if any(indicator in user_lower for indicator in chat_indicators):
            return "chat"
        
        return "general"
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词（增强版）"""
        words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text.lower())
        
        # 中文停用词（增强版）
        stop_words = {
            "的", "了", "和", "是", "在", "我", "有", "你", "他", "她", 
            "它", "这", "那", "就", "都", "也", "不", "吗", "呢", "啊",
            "呀", "吧", "嗯", "哦", "哈", "啦", "哇", "嘛", "哟", "哼"
        }
        
        # 计算词频
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words and len(word) > 1:
                word_freq[word] += 1
        
        # 按频率排序，取前10
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [word for word, _ in keywords]
    
    def detect_emotion(self, text: str) -> str:
        """检测情绪（增强版）"""
        positive_words = {
            "好", "开心", "快乐", "高兴", "谢谢", "喜欢", "爱", "棒", "完美",
            "优秀", "厉害", "强大", "美好", "幸福", "满意", "赞", "棒棒", "太棒了"
        }
        negative_words = {
            "不好", "生气", "难过", "伤心", "讨厌", "恨", "糟糕", "坏", "烦",
            "垃圾", "讨厌", "愤怒", "失望", "悲伤", "痛苦", "难受", "可恶"
        }
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # 考虑强度
        strong_positive = {"爱", "太棒了", "完美", "优秀"}
        strong_negative = {"恨", "垃圾", "可恶", "愤怒"}
        
        for word in strong_positive:
            if word in text_lower:
                pos_count += 2
        
        for word in strong_negative:
            if word in text_lower:
                neg_count += 2
        
        if pos_count > neg_count * 2:
            return "very_positive"
        elif pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count * 2:
            return "very_negative"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def extract_knowledge(self, question: str, answer: str, keywords: List[str], 
                          concepts: List[str], conversation_type: str):
        """提取知识（增强版）"""
        # 如果回答包含事实性信息，保存为知识
        factual_indicators = ["是", "有", "可以", "能够", "会", "要", "需要", "必须", 
                             "应该", "一定", "通常", "一般", "总是", "从不"]
        
        has_factual = any(indicator in answer for indicator in factual_indicators)
        is_teaching = conversation_type == "teaching"
        
        if has_factual or is_teaching:
            # 提取可能的定义
            definition = None
            if "是" in answer:
                parts = answer.split("是", 1)
                if len(parts) == 2:
                    definition = parts[1].strip()
            
            knowledge_data = {
                "question": question,
                "answer": answer,
                "keywords": keywords,
                "concepts": concepts,
                "definition": definition,
                "type": "fact" if has_factual else "teaching",
                "conversation_type": conversation_type
            }
            
            # 生成知识键
            knowledge_key = hashlib.md5((question + answer).encode()).hexdigest()[:12]
            
            # 设置重要性
            importance = 0.8 if is_teaching else 0.7
            
            self.remember(
                category="knowledge",
                key=knowledge_key,
                data=knowledge_data,
                metadata={
                    "importance": importance,
                    "source": "conversation_extraction",
                    "extracted_from": f"conv_{hashlib.md5(question.encode()).hexdigest()[:8]}"
                },
                tags=keywords + ["knowledge", "extracted"]
            )
    
    def update_short_term_weights(self, memory_key: str, importance: float):
        """更新短期记忆权重"""
        # 新记忆初始权重
        self.short_term_weights[memory_key] = 1.0
        
        # 重要记忆权重更高
        if importance > 0.7:
            self.short_term_weights[memory_key] = 2.0
        
        # 衰减旧记忆权重
        for key in list(self.short_term_weights.keys()):
            if key != memory_key:
                self.short_term_weights[key] *= 0.9
                
                # 移除权重过低的记忆
                if self.short_term_weights[key] < 0.1:
                    del self.short_term_weights[key]
    
    def get_short_term_memory(self, limit: int = 10) -> List[Dict]:
        """获取短期记忆（加权）"""
        memories = list(self.short_term_memory)
        
        if not memories:
            return []
        
        # 按权重排序
        weighted_memories = []
        for memory in memories:
            key = memory.get("key", "")
            weight = self.short_term_weights.get(key, 0.5)
            weighted_memories.append((weight, memory))
        
        weighted_memories.sort(reverse=True)
        
        return [memory for _, memory in weighted_memories[:limit]]
    
    def consolidate_memories(self):
        """巩固记忆"""
        # 处理巩固队列
        while self.consolidation_queue:
            memory_id = self.consolidation_queue.popleft()
            
            # 重新加载和保存重要记忆
            if '_' in memory_id:
                parts = memory_id.split('_', 1)
                if len(parts) == 2:
                    category, key = parts
                    data = self.compressed_memory.load_memory(memory_id)
                    if data:
                        # 提高重要性
                        cache_entry = self.compressed_memory.memory_cache.get(key)
                        if cache_entry:
                            new_importance = min(1.0, cache_entry.get("importance", 0.5) + 0.1)
                            cache_entry["importance"] = new_importance
                            
                            # 重新保存
                            metadata = cache_entry.get("metadata", {})
                            metadata["importance"] = new_importance
                            metadata["consolidated"] = True
                            metadata["consolidated_at"] = datetime.now().isoformat()
                            
                            self.remember(category, key, data, metadata)
        
        # 压缩内存的巩固
        self.compressed_memory.consolidate_important_memories()
        
        # 保存概念网络
        self.save_concept_network()
    
    def cleanup(self, days_old: int = 30):
        """清理旧记忆"""
        deleted = self.compressed_memory.cleanup_old_memories(days_old, keep_important=True)
        
        # 清理短期记忆
        cutoff = datetime.now() - timedelta(days=min(7, days_old))
        old_count = 0
        
        for memory in list(self.short_term_memory):
            timestamp = memory.get("timestamp", "")
            if timestamp:
                try:
                    mem_time = datetime.fromisoformat(timestamp)
                    if mem_time < cutoff:
                        self.short_term_memory.remove(memory)
                        old_count += 1
                except:
                    pass
        
        print(f"清理了 {old_count} 条旧短期记忆")
        
        # 执行巩固
        self.consolidate_memories()
        
        return deleted + old_count
    
    def get_stats(self) -> Dict:
        """获取记忆统计（增强版）"""
        memory_stats = self.compressed_memory.get_stats()
        
        # 短期记忆统计
        short_term_by_type = defaultdict(int)
        for memory in self.short_term_memory:
            if isinstance(memory.get("data"), dict):
                conv_type = memory["data"].get("type", "unknown")
                short_term_by_type[conv_type] += 1
        
        return {
            **memory_stats,
            "categories": {cat: len(ids) for cat, ids in self.memory_index.items()},
            "short_term_count": len(self.short_term_memory),
            "short_term_by_type": dict(short_term_by_type),
            "total_memories": sum(len(ids) for ids in self.memory_index.values()),
            "concept_network_size": len(self.concept_network),
            "semantic_links_count": sum(len(links) for links in self.semantic_links.values()),
            "memory_categories": len(self.memory_categories)
        }

def main():
    """记忆系统测试"""
    print("测试增强版记忆系统...")
    
    # 创建配置
    from config import AIConfig
    config = AIConfig()
    
    # 创建记忆系统
    memory = IntelligentMemory(config)
    
    # 测试记忆
    test_data = {
        "name": "测试用户",
        "preference": "喜欢蓝色和绿色",
        "knowledge": "Python是一种编程语言",
        "last_visited": datetime.now().isoformat()
    }
    
    # 保存记忆
    success = memory.remember(
        category="preference",
        key="user_prefs_v2",
        data=test_data,
        metadata={"importance": 0.9, "context": "用户偏好和知识"},
        tags=["user", "preference", "knowledge", "python"]
    )
    
    print(f"保存记忆: {'成功' if success else '失败'}")
    
    # 回忆记忆
    recalled = memory.recall(category="preference", key="user_prefs_v2")
    print(f"回忆记忆: {len(recalled)} 条")
    
    # 测试带排序的回忆
    recalled_sorted = memory.recall(limit=5, sort_by="recent")
    print(f"按时间排序回忆: {len(recalled_sorted)} 条")
    
    # 搜索记忆
    search_results = memory.search("Python")
    print(f"搜索结果: {len(search_results)} 条")
    
    # 获取统计
    stats = memory.get_stats()
    print(f"记忆统计:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # 测试对话学习
    memory.learn_from_conversation(
        "Python是什么？",
        "Python是一种高级编程语言，以简洁易读著称。",
        {"topic": "编程", "difficulty": "入门"}
    )
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()