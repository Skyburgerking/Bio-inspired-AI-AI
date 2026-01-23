import cv2
import numpy as np
import struct
import threading
import time
import os
import sys
import json
import socket
import pickle
import queue
import ctypes
import platform
import subprocess
import random
import hashlib
import sqlite3
import base64
from collections import deque
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
import math
from scipy import ndimage
from scipy.spatial import distance

# ==================== Windows权限检测与处理 ====================

def is_admin():
    """检查当前是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def request_admin_privileges():
    """请求管理员权限重新运行程序"""
    if platform.system() != "Windows":
        return False
    
    if not is_admin():
        print("检测到需要管理员权限进行屏幕捕获...")
        print("正在请求管理员权限...")
        
        # 获取当前脚本路径
        script = sys.executable
        params = ' '.join([f'"{arg}"' for arg in sys.argv])
        
        # 使用ShellExecute以管理员权限重新运行
        try:
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", script, params, None, 1
            )
            print("已请求管理员权限，程序将以管理员身份重新启动...")
            sys.exit(0)
            return True
        except Exception as e:
            print(f"请求管理员权限失败: {e}")
            return False
    
    return True

# ==================== 第三方库导入与检查 ====================

def import_required_libraries():
    """导入所需的第三方库"""
    required_libs = {
        'opencv-python': 'cv2',
        'scikit-learn': 'sklearn',
        'numpy': 'np',
        'joblib': 'joblib',
        'pillow': 'PIL',
        'pyautogui': 'pyautogui',
        'screeninfo': 'screeninfo',
        'scipy': 'scipy'
    }
    
    missing_libs = []
    
    for lib_name, import_name in required_libs.items():
        try:
            if import_name == 'cv2':
                import cv2
            elif import_name == 'np':
                import numpy as np
            elif import_name == 'sklearn':
                import sklearn
            elif import_name == 'PIL':
                from PIL import Image
            elif import_name == 'screeninfo':
                import screeninfo
            elif import_name == 'scipy':
                import scipy
            else:
                __import__(import_name)
        except ImportError:
            missing_libs.append(lib_name)
    
    if missing_libs:
        print(f"缺少以下库: {', '.join(missing_libs)}")
        print("正在尝试安装...")
        
        try:
            import pip
            for lib in missing_libs:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                    print(f"成功安装: {lib}")
                except subprocess.CalledProcessError:
                    print(f"安装 {lib} 失败，请手动安装")
            
            print("请重新启动程序")
            sys.exit(1)
            
        except Exception as e:
            print(f"安装库时出错: {e}")
            print("请手动运行: pip install opencv-python scikit-learn numpy joblib pillow pyautogui screeninfo scipy")
            sys.exit(1)
    
    print("所有必需库导入成功")

# 在程序开始时检查
import_required_libraries()

# 现在导入所有需要的库
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
from PIL import ImageGrab
import pyautogui
import screeninfo
from scipy import ndimage
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

# ==================== GEDA 相关定义 ====================

class BaseType(Enum):
    """五进制碱基类型"""
    AGGRESSIVE = 'A'  # 激进行动
    STABLE = 'G'      # 稳定操作
    CONSERVATIVE = 'C' # 保守策略
    FLEXIBLE = 'T'    # 灵活调整
    MUTANT = 'X'      # 突变探索

class ThreatLevel(Enum):
    """威胁等级"""
    NONE = 0     # 无威胁
    LOW = 1      # 低威胁
    MEDIUM = 2   # 中威胁
    HIGH = 3     # 高威胁
    CRITICAL = 4 # 致命威胁

class MaterialType(Enum):
    """材料类型"""
    UNKNOWN = 0      # 未知
    METAL = 1        # 金属
    WOOD = 2         # 木材
    STONE = 3        # 石材
    PLASTIC = 4      # 塑料
    TEXTILE = 5      # 纺织品
    GLASS = 6        # 玻璃
    LIQUID = 7       # 液体
    SKIN = 8         # 皮肤/生物组织
    VEGETATION = 9   # 植被
    SKY = 10         # 天空/背景

# ==================== 色彩与材料科学类 ====================

class ColorMaterialAnalyzer:
    """色彩与材料分析器"""
    
    def __init__(self):
        # 材料色彩数据库 (BGR格式)
        self.material_color_profiles = {
            MaterialType.METAL: {
                'primary_colors': [
                    np.array([200, 200, 220]),  # 银白色
                    np.array([180, 180, 200]),  # 钢铁灰
                    np.array([100, 100, 150])   # 深金属色
                ],
                'reflectivity': 0.7,
                'roughness': 0.3,
                'grayscale_range': (150, 220)
            },
            MaterialType.WOOD: {
                'primary_colors': [
                    np.array([70, 100, 150]),   # 深棕色
                    np.array([50, 80, 120]),    # 红木色
                    np.array([30, 50, 80])      # 深木色
                ],
                'reflectivity': 0.3,
                'roughness': 0.8,
                'grayscale_range': (50, 120)
            },
            MaterialType.STONE: {
                'primary_colors': [
                    np.array([150, 150, 150]),  # 灰色
                    np.array([120, 120, 120]),  # 深灰色
                    np.array([180, 180, 180])   # 浅灰色
                ],
                'reflectivity': 0.4,
                'roughness': 0.7,
                'grayscale_range': (100, 180)
            },
            MaterialType.VEGETATION: {
                'primary_colors': [
                    np.array([50, 150, 50]),    # 绿色
                    np.array([30, 120, 30]),    # 深绿色
                    np.array([80, 180, 80])     # 浅绿色
                ],
                'reflectivity': 0.2,
                'roughness': 0.9,
                'grayscale_range': (80, 150)
            },
            MaterialType.SKY: {
                'primary_colors': [
                    np.array([255, 200, 100]),  # 天空蓝
                    np.array([200, 150, 50]),   # 深天蓝
                    np.array([150, 100, 30])    # 黄昏色
                ],
                'reflectivity': 0.1,
                'roughness': 0.1,
                'grayscale_range': (100, 200)
            }
        }
        
        # 颜色聚类参数
        self.color_cluster_epsilon = 15
        self.min_cluster_samples = 10
        
        # 缓存系统
        self.color_cache = {}
        self.cache_max_size = 100
        
    def analyze_color_material(self, image_region: np.ndarray) -> Dict[str, Any]:
        """
        分析区域的颜色和材料属性
        
        Args:
            image_region: 图像区域 (BGR格式)
            
        Returns:
            材料分析结果字典
        """
        if image_region is None or image_region.size == 0:
            return self._get_default_material()
        
        # 生成缓存键
        cache_key = self._generate_cache_key(image_region)
        if cache_key in self.color_cache:
            return self.color_cache[cache_key]
        
        # 1. 提取主要颜色
        dominant_colors = self._extract_dominant_colors(image_region)
        
        # 2. 分析颜色分布
        color_distribution = self._analyze_color_distribution(image_region)
        
        # 3. 计算灰度特性
        grayscale_properties = self._analyze_grayscale(image_region)
        
        # 4. 材料类型识别
        material_type, material_confidence = self._identify_material(
            dominant_colors, color_distribution, grayscale_properties
        )
        
        # 5. 计算粘合度
        cohesion = self._calculate_cohesion(image_region)
        
        # 6. 光影特征分析
        shadow_features = self._analyze_shadow_features(image_region)
        
        # 7. 透视特征
        perspective_features = self._analyze_perspective(image_region)
        
        result = {
            'material_type': material_type.name,
            'material_confidence': material_confidence,
            'dominant_colors': [color.tolist() for color in dominant_colors],
            'color_variance': color_distribution['variance'],
            'grayscale_mean': grayscale_properties['mean'],
            'grayscale_std': grayscale_properties['std'],
            'cohesion': cohesion,
            'has_shadow': shadow_features['has_shadow'],
            'shadow_intensity': shadow_features['intensity'],
            'perspective_score': perspective_features['score'],
            'texture_complexity': self._calculate_texture_complexity(image_region)
        }
        
        # 缓存结果
        self._update_cache(cache_key, result)
        
        return result
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[np.ndarray]:
        """提取主要颜色 (K-means聚类简化版)"""
        # 转换为像素列表
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # 简单聚类：使用颜色直方图峰值
        color_hist = {}
        for pixel in pixels[:1000]:  # 采样
            color_key = tuple((pixel / 10).astype(int) * 10)  # 量化
            color_hist[color_key] = color_hist.get(color_key, 0) + 1
        
        # 取前n个颜色
        sorted_colors = sorted(color_hist.items(), key=lambda x: x[1], reverse=True)[:n_colors]
        dominant_colors = [np.array(color[0]) for color in sorted_colors]
        
        return dominant_colors
    
    def _analyze_color_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """分析颜色分布"""
        # 计算颜色方差
        pixels = image.reshape(-1, 3)
        mean_color = np.mean(pixels, axis=0)
        variance = np.mean(np.var(pixels, axis=0))
        
        # 计算颜色相似度
        unique_ratio = len(np.unique(pixels, axis=0)) / len(pixels)
        
        return {
            'variance': float(variance),
            'unique_ratio': float(unique_ratio),
            'mean_color': mean_color.tolist()
        }
    
    def _analyze_grayscale(self, image: np.ndarray) -> Dict[str, float]:
        """分析灰度特性"""
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算统计特性
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # 计算灰度直方图特性
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # 计算熵（复杂度）
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'entropy': float(entropy)
        }
    
    def _identify_material(self, dominant_colors: List[np.ndarray], 
                          color_distribution: Dict[str, float], 
                          grayscale_properties: Dict[str, float]) -> Tuple[MaterialType, float]:
        """识别材料类型"""
        best_match = MaterialType.UNKNOWN
        best_confidence = 0.0
        
        gray_mean = grayscale_properties['mean']
        color_var = color_distribution['variance']
        
        # 基于颜色和灰度特征匹配材料
        for material_type, profile in self.material_color_profiles.items():
            # 检查灰度范围
            gray_min, gray_max = profile['grayscale_range']
            if not (gray_min <= gray_mean <= gray_max):
                continue
            
            # 颜色相似度计算
            color_similarities = []
            for sample_color in dominant_colors[:3]:
                if len(sample_color) != 3:
                    continue
                    
                for ref_color in profile['primary_colors']:
                    # 计算颜色距离
                    color_distance = np.linalg.norm(sample_color - ref_color)
                    similarity = max(0, 1 - color_distance / 255.0)
                    color_similarities.append(similarity)
            
            if not color_similarities:
                continue
                
            avg_similarity = np.mean(color_similarities)
            
            # 综合考虑颜色相似度和颜色方差
            # 某些材料应有较低的颜色方差（如金属），某些应有较高的方差（如植被）
            expected_var = 50  # 默认值
            var_similarity = 1.0 / (1.0 + abs(color_var - expected_var) / 100.0)
            
            confidence = avg_similarity * 0.7 + var_similarity * 0.3
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = material_type
        
        return best_match, best_confidence
    
    def _calculate_cohesion(self, image: np.ndarray) -> float:
        """
        计算区域粘合度（颜色/纹理一致性）
        粘合度高表示区域内部一致性强，适合整体处理
        """
        if image.size == 0:
            return 0.0
        
        # 1. 颜色一致性
        pixels = image.reshape(-1, 3)
        color_std = np.mean(np.std(pixels, axis=0))
        color_cohesion = 1.0 / (1.0 + color_std / 30.0)  # 标准化到0-1
        
        # 2. 纹理一致性（通过边缘检测）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 低边缘密度表示纹理一致
        texture_cohesion = 1.0 - min(edge_density * 5, 1.0)
        
        # 3. 梯度一致性
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_variance = np.var(grad_magnitude)
        grad_cohesion = 1.0 / (1.0 + grad_variance / 10000.0)
        
        # 综合粘合度
        cohesion = (color_cohesion * 0.4 + texture_cohesion * 0.3 + grad_cohesion * 0.3)
        
        return float(cohesion)
    
    def _analyze_shadow_features(self, image: np.ndarray) -> Dict[str, Any]:
        """分析阴影特征"""
        if image.size == 0:
            return {'has_shadow': False, 'intensity': 0.0, 'direction': 0.0}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用光照估计技术
        # 1. 计算梯度方向直方图
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        if grad_x.size == 0 or grad_y.size == 0:
            return {'has_shadow': False, 'intensity': 0.0, 'direction': 0.0}
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # 2. 检测阴影区域（低亮度+高梯度）
        mean_brightness = np.mean(gray)
        shadow_mask = (gray < mean_brightness * 0.7) & (magnitude > np.mean(magnitude) * 0.5)
        
        has_shadow = np.sum(shadow_mask) > shadow_mask.size * 0.05  # 5%以上阴影区域
        
        # 3. 阴影强度
        shadow_intensity = 0.0
        if has_shadow:
            shadow_intensity = np.mean(magnitude[shadow_mask]) / 255.0
        
        return {
            'has_shadow': bool(has_shadow),
            'intensity': float(shadow_intensity),
            'direction': float(np.mean(direction) if np.any(shadow_mask) else 0.0)
        }
    
    def _analyze_perspective(self, image: np.ndarray) -> Dict[str, float]:
        """分析透视特征"""
        if image.size == 0:
            return {'score': 0.0, 'vanishing_point': None}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测边缘
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测直线（霍夫变换）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=30, maxLineGap=10)
        
        perspective_score = 0.0
        vanishing_point = None
        
        if lines is not None and len(lines) > 2:
            # 分析直线方向分布
            angles = []
            for line in lines[:20]:  # 限制数量
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)
            
            # 计算角度方差（低方差表示平行线，高方差表示透视）
            if angles:
                angle_var = np.var(angles)
                perspective_score = min(angle_var / (np.pi/4), 1.0)  # 归一化到0-1
        
        return {
            'score': float(perspective_score),
            'vanishing_point': vanishing_point
        }
    
    def _calculate_texture_complexity(self, image: np.ndarray) -> float:
        """计算纹理复杂度"""
        if image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用LBP（局部二值模式）计算纹理复杂度
        radius = 1
        n_points = 8 * radius
        
        # 简化版LBP计算
        height, width = gray.shape
        texture_map = np.zeros_like(gray)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = gray[i, j]
                code = 0
                
                # 采样周围点
                for n in range(n_points):
                    angle = 2 * np.pi * n / n_points
                    x = int(j + radius * np.cos(angle))
                    y = int(i - radius * np.sin(angle))
                    
                    if gray[y, x] >= center:
                        code |= (1 << n)
                
                texture_map[i, j] = code
        
        # 计算纹理复杂度（不同LBP模式的数量）
        unique_codes = len(np.unique(texture_map))
        complexity = min(unique_codes / 256.0, 1.0)
        
        return float(complexity)
    
    def _generate_cache_key(self, image: np.ndarray) -> str:
        """生成缓存键（基于图像哈希）"""
        # 计算简单哈希
        small_img = cv2.resize(image, (8, 8))
        avg_color = np.mean(small_img)
        diff = small_img > avg_color
        hash_str = ''.join(['1' if x else '0' for x in diff.flatten()])
        return hash_str[:64]
    
    def _update_cache(self, key: str, value: Dict[str, Any]):
        """更新缓存"""
        if len(self.color_cache) >= self.cache_max_size:
            # 移除最旧的条目
            oldest_key = next(iter(self.color_cache))
            del self.color_cache[oldest_key]
        
        self.color_cache[key] = value
    
    def _get_default_material(self) -> Dict[str, Any]:
        """获取默认材料结果"""
        return {
            'material_type': MaterialType.UNKNOWN.name,
            'material_confidence': 0.0,
            'dominant_colors': [],
            'color_variance': 0.0,
            'grayscale_mean': 0.0,
            'grayscale_std': 0.0,
            'cohesion': 0.0,
            'has_shadow': False,
            'shadow_intensity': 0.0,
            'perspective_score': 0.0,
            'texture_complexity': 0.0
        }

# ==================== 光影结构分析器 ====================

class LightShadowAnalyzer:
    """光影结构分析器"""
    
    def __init__(self):
        self.shadow_detector = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.light_sources = []
        
    def analyze_light_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析光影结构
        
        Args:
            image: 输入图像
            
        Returns:
            光影分析结果
        """
        if image is None or image.size == 0:
            return self._get_default_light_analysis()
        
        # 1. 去色处理，专注光影结构
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. 使用Retinex算法增强光照不变性
        illumination_invariant = self._apply_retinex(gray)
        
        # 3. 检测阴影区域
        shadow_mask = self._detect_shadows(gray, illumination_invariant)
        
        # 4. 估计光照方向
        light_direction = self._estimate_light_direction(gray, shadow_mask)
        
        # 5. 分析高光和反光
        highlight_features = self._analyze_highlights(image)
        
        # 6. 计算光影对比度
        contrast_metrics = self._calculate_light_contrast(gray)
        
        # 7. 恢复颜色光影关系
        color_shadow_relation = self._analyze_color_shadow_relation(image, shadow_mask)
        
        return {
            'illumination_consistency': float(np.mean(illumination_invariant)),
            'shadow_coverage': float(np.sum(shadow_mask) / shadow_mask.size),
            'light_direction': light_direction,
            'highlight_count': highlight_features['count'],
            'highlight_intensity': highlight_features['intensity'],
            'light_contrast': contrast_metrics['contrast'],
            'light_gradient': contrast_metrics['gradient'],
            'color_shadow_correlation': color_shadow_relation['correlation'],
            'has_consistent_lighting': self._check_consistent_lighting(gray, shadow_mask)
        }
    
    def _apply_retinex(self, gray_image: np.ndarray) -> np.ndarray:
        """应用Retinex算法提取光照不变特征"""
        # 简化的单尺度Retinex
        log_image = np.log1p(gray_image.astype(np.float32))
        
        # 高斯模糊作为光照估计
        sigma = 30
        illumination = gaussian_filter(gray_image.astype(np.float32), sigma)
        log_illumination = np.log1p(illumination + 1e-10)
        
        # 反射分量（光照不变）
        reflectance = log_image - log_illumination
        
        # 归一化
        reflectance = (reflectance - np.min(reflectance)) / (np.max(reflectance) - np.min(reflectance) + 1e-10)
        
        return reflectance
    
    def _detect_shadows(self, gray_image: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
        """检测阴影区域"""
        # 方法1：基于局部对比度
        local_mean = cv2.blur(gray_image, (15, 15))
        local_std = cv2.blur((gray_image - local_mean)**2, (15, 15))**0.5
        
        # 低亮度+低对比度区域可能是阴影
        shadow_candidate1 = (gray_image < np.mean(gray_image) * 0.7) & (local_std < np.mean(local_std) * 0.8)
        
        # 方法2：基于Retinex反射率
        shadow_candidate2 = reflectance < np.percentile(reflectance, 30)
        
        # 合并阴影检测
        shadow_mask = shadow_candidate1 | shadow_candidate2
        
        # 形态学操作清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask > 0
    
    def _estimate_light_direction(self, gray_image: np.ndarray, shadow_mask: np.ndarray) -> Dict[str, float]:
        """估计光照方向"""
        # 使用梯度分析
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算平均梯度方向（避开阴影区域）
        if np.any(~shadow_mask):
            valid_grad_x = grad_x[~shadow_mask]
            valid_grad_y = grad_y[~shadow_mask]
            
            if len(valid_grad_x) > 0:
                mean_grad_x = np.mean(valid_grad_x)
                mean_grad_y = np.mean(valid_grad_y)
                
                # 光照方向与梯度方向相反（光照从亮到暗）
                direction_angle = np.arctan2(-mean_grad_y, -mean_grad_x)
                
                return {
                    'angle_radians': float(direction_angle),
                    'angle_degrees': float(np.degrees(direction_angle)),
                    'strength': float(np.sqrt(mean_grad_x**2 + mean_grad_y**2))
                }
        
        return {'angle_radians': 0.0, 'angle_degrees': 0.0, 'strength': 0.0}
    
    def _analyze_highlights(self, image: np.ndarray) -> Dict[str, Any]:
        """分析高光区域"""
        # 转换到HSV空间检测高光
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 高光区域：高亮度+低饱和度
        highlight_mask = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 50)
        
        # 计算高光特性
        highlight_count = np.sum(highlight_mask)
        if highlight_count > 0:
            highlight_intensity = np.mean(hsv[highlight_mask, 2])
        else:
            highlight_intensity = 0
        
        return {
            'count': int(highlight_count),
            'intensity': float(highlight_intensity),
            'coverage': float(highlight_count / highlight_mask.size)
        }
    
    def _calculate_light_contrast(self, gray_image: np.ndarray) -> Dict[str, float]:
        """计算光照对比度"""
        # 局部对比度
        local_contrast = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        # 全局对比度（直方图拉伸）
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        
        # 计算熵
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 计算梯度幅值
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(grad_magnitude)
        
        return {
            'contrast': float(local_contrast / 10000.0),  # 归一化
            'entropy': float(entropy / 8.0),  # 归一化到0-1
            'gradient': float(avg_gradient / 255.0)  # 归一化
        }
    
    def _analyze_color_shadow_relation(self, image: np.ndarray, shadow_mask: np.ndarray) -> Dict[str, float]:
        """分析颜色与阴影的关系"""
        if not np.any(shadow_mask) or not np.any(~shadow_mask):
            return {'correlation': 0.0, 'color_shift': 0.0}
        
        # 分离阴影区和非阴影区的颜色
        shadow_colors = image[shadow_mask].reshape(-1, 3)
        light_colors = image[~shadow_mask].reshape(-1, 3)
        
        if len(shadow_colors) == 0 or len(light_colors) == 0:
            return {'correlation': 0.0, 'color_shift': 0.0}
        
        # 计算平均颜色差异
        mean_shadow = np.mean(shadow_colors, axis=0)
        mean_light = np.mean(light_colors, axis=0)
        
        # 颜色偏移（阴影区通常更暗、更蓝）
        color_shift = np.linalg.norm(mean_shadow - mean_light) / 441.67  # 归一化到0-1
        
        # 计算颜色相关性
        # 简化：计算亮度通道的相关性
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shadow_brightness = gray[shadow_mask]
        light_brightness = gray[~shadow_mask]
        
        # 采样以减少计算量
        sample_size = min(100, len(shadow_brightness), len(light_brightness))
        if sample_size > 10:
            shadow_sample = np.random.choice(shadow_brightness, sample_size, replace=False)
            light_sample = np.random.choice(light_brightness, sample_size, replace=False)
            
            # 计算相关系数
            correlation = np.corrcoef(shadow_sample, light_sample)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'correlation': float(correlation),
            'color_shift': float(color_shift)
        }
    
    def _check_consistent_lighting(self, gray_image: np.ndarray, shadow_mask: np.ndarray) -> bool:
        """检查光照一致性"""
        # 计算非阴影区域的亮度方差
        if np.any(~shadow_mask):
            light_areas = gray_image[~shadow_mask]
            brightness_variance = np.var(light_areas)
            
            # 低方差表示一致光照
            return brightness_variance < 1000  # 阈值可调
        return False
    
    def _get_default_light_analysis(self) -> Dict[str, Any]:
        """获取默认光影分析结果"""
        return {
            'illumination_consistency': 0.0,
            'shadow_coverage': 0.0,
            'light_direction': {'angle_radians': 0.0, 'angle_degrees': 0.0, 'strength': 0.0},
            'highlight_count': 0,
            'highlight_intensity': 0.0,
            'light_contrast': 0.0,
            'light_gradient': 0.0,
            'color_shadow_correlation': 0.0,
            'has_consistent_lighting': False
        }

# ==================== 宏观计算优化器 ====================

class MacroOptimizer:
    """宏观计算优化器 - 处理自然风景等复杂场景"""
    
    def __init__(self, cohesion_threshold: float = 0.7, max_processing_area: int = 50000):
        """
        初始化宏观优化器
        
        Args:
            cohesion_threshold: 粘合度阈值，超过则视为一致区域
            max_processing_area: 最大处理区域面积（像素）
        """
        self.cohesion_threshold = cohesion_threshold
        self.max_processing_area = max_processing_area
        
        # 区域合并器
        self.region_merger = RegionMerger()
        
        # 缓存系统
        self.region_cache = {}
        self.cache_max_size = 50
        
        # 统计信息
        self.skipped_regions = 0
        self.processed_regions = 0
        
    def optimize_processing(self, frame: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        优化区域处理，合并高粘合度区域，跳过过于复杂的区域
        
        Args:
            frame: 完整帧
            regions: 检测到的区域列表 (x, y, w, h)
            
        Returns:
            优化后的区域列表，包含元数据
        """
        if not regions:
            return []
        
        optimized_regions = []
        
        for region in regions:
            x, y, w, h = region
            
            # 限制区域大小
            if w * h > self.max_processing_area:
                # 区域过大，进行分割
                sub_regions = self._split_large_region(x, y, w, h)
                for sub_region in sub_regions:
                    opt_result = self._process_single_region(frame, sub_region)
                    if opt_result:
                        optimized_regions.append(opt_result)
            else:
                opt_result = self._process_single_region(frame, region)
                if opt_result:
                    optimized_regions.append(opt_result)
        
        # 尝试合并相邻的高粘合度区域
        if len(optimized_regions) > 1:
            optimized_regions = self.region_merger.merge_similar_regions(optimized_regions)
        
        return optimized_regions
    
    def _process_single_region(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        """处理单个区域"""
        x, y, w, h = region
        
        # 提取区域图像
        region_img = frame[y:y+h, x:x+w]
        if region_img.size == 0:
            return None
        
        # 生成缓存键
        cache_key = f"{x},{y},{w},{h}_{hashlib.md5(region_img.tobytes()).hexdigest()[:16]}"
        
        if cache_key in self.region_cache:
            result = self.region_cache[cache_key].copy()
            result['bbox'] = region  # 更新边界框
            return result
        
        # 分析区域特性
        color_analyzer = ColorMaterialAnalyzer()
        material_info = color_analyzer.analyze_color_material(region_img)
        
        # 计算区域复杂度
        complexity = self._calculate_region_complexity(region_img, material_info)
        
        # 决策：是否需要详细处理
        should_process = self._should_process_region(material_info, complexity)
        
        result = {
            'bbox': region,
            'material_info': material_info,
            'complexity': complexity,
            'should_process': should_process,
            'is_natural_scenery': self._is_natural_scenery(material_info),
            'estimated_importance': self._estimate_importance(region_img, material_info)
        }
        
        # 更新统计
        if not should_process:
            self.skipped_regions += 1
        else:
            self.processed_regions += 1
        
        # 缓存结果
        self._update_region_cache(cache_key, result)
        
        return result
    
    def _split_large_region(self, x: int, y: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """分割大区域"""
        max_sub_size = int(np.sqrt(self.max_processing_area))
        
        sub_regions = []
        rows = (h + max_sub_size - 1) // max_sub_size
        cols = (w + max_sub_size - 1) // max_sub_size
        
        sub_h = h // rows
        sub_w = w // cols
        
        for i in range(rows):
            for j in range(cols):
                sub_x = x + j * sub_w
                sub_y = y + i * sub_h
                sub_width = sub_w if j < cols - 1 else w - j * sub_w
                sub_height = sub_h if i < rows - 1 else h - i * sub_h
                
                if sub_width > 0 and sub_height > 0:
                    sub_regions.append((sub_x, sub_y, sub_width, sub_height))
        
        return sub_regions
    
    def _calculate_region_complexity(self, region_img: np.ndarray, material_info: Dict[str, Any]) -> float:
        """计算区域复杂度"""
        # 基于多个因素计算复杂度
        factors = []
        
        # 1. 颜色复杂度
        color_variance = material_info.get('color_variance', 0.0)
        factors.append(min(color_variance / 100.0, 1.0))
        
        # 2. 纹理复杂度
        texture_complexity = material_info.get('texture_complexity', 0.0)
        factors.append(texture_complexity)
        
        # 3. 边缘密度
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        factors.append(min(edge_density * 5, 1.0))
        
        # 4. 材料类型复杂度
        material_type = material_info.get('material_type', 'UNKNOWN')
        material_complexity = {
            'UNKNOWN': 0.5,
            'METAL': 0.3,
            'WOOD': 0.4,
            'STONE': 0.4,
            'PLASTIC': 0.3,
            'TEXTILE': 0.6,
            'GLASS': 0.7,
            'LIQUID': 0.8,
            'SKIN': 0.5,
            'VEGETATION': 0.7,
            'SKY': 0.1
        }.get(material_type, 0.5)
        factors.append(material_complexity)
        
        # 综合复杂度
        complexity = np.mean(factors)
        return float(complexity)
    
    def _should_process_region(self, material_info: Dict[str, Any], complexity: float) -> bool:
        """决定是否需要详细处理区域"""
        cohesion = material_info.get('cohesion', 0.0)
        material_type = material_info.get('material_type', 'UNKNOWN')
        
        # 高粘合度区域可以简化处理
        if cohesion > self.cohesion_threshold:
            # 除非是重要材料类型
            if material_type in ['SKIN', 'METAL', 'GLASS']:
                return True
            else:
                return False
        
        # 低复杂度区域可以简化
        if complexity < 0.3:
            return False
        
        # 高复杂度区域需要处理
        if complexity > 0.7:
            return True
        
        # 中等复杂度：基于材料类型决定
        important_materials = ['SKIN', 'METAL', 'GLASS', 'TEXTILE']
        return material_type in important_materials
    
    def _is_natural_scenery(self, material_info: Dict[str, Any]) -> bool:
        """判断是否为自然风景"""
        material_type = material_info.get('material_type', 'UNKNOWN')
        natural_materials = ['VEGETATION', 'SKY', 'STONE', 'WOOD']
        
        if material_type in natural_materials:
            return True
        
        # 通过颜色特征判断
        dominant_colors = material_info.get('dominant_colors', [])
        if len(dominant_colors) >= 2:
            # 检查是否有绿色或蓝色（自然色）
            for color in dominant_colors[:2]:
                if len(color) == 3:
                    b, g, r = color
                    # 绿色或蓝色占主导
                    if g > r * 1.2 and g > b * 1.2:  # 绿色
                        return True
                    if b > r * 1.2 and b > g * 1.2:  # 蓝色
                        return True
        
        return False
    
    def _estimate_importance(self, region_img: np.ndarray, material_info: Dict[str, Any]) -> float:
        """估计区域重要性"""
        importance = 0.0
        
        # 1. 材料重要性
        material_type = material_info.get('material_type', 'UNKNOWN')
        material_importance = {
            'SKIN': 0.9,
            'METAL': 0.8,
            'GLASS': 0.7,
            'TEXTILE': 0.6,
            'PLASTIC': 0.5,
            'WOOD': 0.4,
            'STONE': 0.3,
            'VEGETATION': 0.2,
            'SKY': 0.1,
            'LIQUID': 0.6,
            'UNKNOWN': 0.5
        }.get(material_type, 0.5)
        importance += material_importance * 0.3
        
        # 2. 位置重要性（中心区域更重要）
        height, width = region_img.shape[:2]
        center_x, center_y = width // 2, height // 2
        distance_to_center = np.sqrt((center_x - width/2)**2 + (center_y - height/2)**2)
        max_distance = np.sqrt((width/2)**2 + (height/2)**2)
        position_importance = 1.0 - (distance_to_center / max_distance)
        importance += position_importance * 0.2
        
        # 3. 大小重要性
        size_importance = min(region_img.size / 10000.0, 1.0)
        importance += size_importance * 0.2
        
        # 4. 边缘强度重要性
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_strength = np.sum(edges > 0) / edges.size
        importance += edge_strength * 0.3
        
        return float(min(importance, 1.0))
    
    def _update_region_cache(self, key: str, value: Dict[str, Any]):
        """更新区域缓存"""
        if len(self.region_cache) >= self.cache_max_size:
            # 移除最旧的条目
            oldest_key = next(iter(self.region_cache))
            del self.region_cache[oldest_key]
        
        self.region_cache[key] = value
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'skipped_regions': self.skipped_regions,
            'processed_regions': self.processed_regions,
            'skipped_ratio': self.skipped_regions / max(self.skipped_regions + self.processed_regions, 1),
            'cache_size': len(self.region_cache)
        }

# ==================== 区域合并器 ====================

class RegionMerger:
    """区域合并器 - 合并相似区域"""
    
    def __init__(self, similarity_threshold: float = 0.6, max_distance: int = 50):
        self.similarity_threshold = similarity_threshold
        self.max_distance = max_distance
    
    def merge_similar_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并相似区域"""
        if len(regions) <= 1:
            return regions
        
        merged_regions = []
        used_indices = set()
        
        for i in range(len(regions)):
            if i in used_indices:
                continue
            
            current_region = regions[i]
            merge_candidates = [current_region]
            
            # 寻找相似区域
            for j in range(i + 1, len(regions)):
                if j in used_indices:
                    continue
                
                other_region = regions[j]
                
                # 检查相似性
                if self._are_regions_similar(current_region, other_region):
                    merge_candidates.append(other_region)
                    used_indices.add(j)
            
            # 合并候选区域
            if len(merge_candidates) > 1:
                merged = self._merge_regions(merge_candidates)
                merged_regions.append(merged)
            else:
                merged_regions.append(current_region)
            
            used_indices.add(i)
        
        return merged_regions
    
    def _are_regions_similar(self, region1: Dict[str, Any], region2: Dict[str, Any]) -> bool:
        """检查两个区域是否相似"""
        # 1. 空间距离
        bbox1 = region1['bbox']
        bbox2 = region2['bbox']
        distance = self._bbox_distance(bbox1, bbox2)
        
        if distance > self.max_distance:
            return False
        
        # 2. 材料相似性
        mat1 = region1.get('material_info', {})
        mat2 = region2.get('material_info', {})
        
        if mat1.get('material_type') != mat2.get('material_type'):
            return False
        
        # 3. 颜色相似性
        color_sim = self._color_similarity(mat1, mat2)
        if color_sim < 0.7:
            return False
        
        # 4. 粘合度相似性
        cohesion1 = mat1.get('cohesion', 0.0)
        cohesion2 = mat2.get('cohesion', 0.0)
        if abs(cohesion1 - cohesion2) > 0.3:
            return False
        
        return True
    
    def _bbox_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的距离"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算中心点距离
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance
    
    def _color_similarity(self, mat1: Dict[str, Any], mat2: Dict[str, Any]) -> float:
        """计算颜色相似性"""
        colors1 = mat1.get('dominant_colors', [])
        colors2 = mat2.get('dominant_colors', [])
        
        if not colors1 or not colors2:
            return 0.0
        
        # 比较主要颜色
        similarity_sum = 0.0
        count = 0
        
        for c1 in colors1[:2]:  # 取前两个主要颜色
            if len(c1) != 3:
                continue
                
            best_similarity = 0.0
            for c2 in colors2[:2]:
                if len(c2) != 3:
                    continue
                
                # 计算颜色距离
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                similarity = max(0, 1 - dist / 441.67)  # 最大距离为sqrt(255^2*3)
                best_similarity = max(best_similarity, similarity)
            
            similarity_sum += best_similarity
            count += 1
        
        return similarity_sum / max(count, 1)
    
    def _merge_regions(self, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个区域"""
        if not regions:
            return {}
        
        # 合并边界框
        bboxes = [r['bbox'] for r in regions]
        merged_bbox = self._merge_bboxes(bboxes)
        
        # 合并材料信息
        material_infos = [r.get('material_info', {}) for r in regions]
        merged_material = self._merge_material_info(material_infos)
        
        # 合并其他属性
        merged_complexity = np.mean([r.get('complexity', 0.0) for r in regions])
        merged_importance = np.mean([r.get('estimated_importance', 0.0) for r in regions])
        
        return {
            'bbox': merged_bbox,
            'material_info': merged_material,
            'complexity': float(merged_complexity),
            'should_process': any(r.get('should_process', False) for r in regions),
            'is_natural_scenery': any(r.get('is_natural_scenery', False) for r in regions),
            'estimated_importance': float(merged_importance),
            'merged_from': len(regions)
        }
    
    def _merge_bboxes(self, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """合并边界框"""
        if not bboxes:
            return (0, 0, 0, 0)
        
        # 找到最小外包矩形
        min_x = min(b[0] for b in bboxes)
        min_y = min(b[1] for b in bboxes)
        max_x = max(b[0] + b[2] for b in bboxes)
        max_y = max(b[1] + b[3] for b in bboxes)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _merge_material_info(self, material_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并材料信息"""
        if not material_infos:
            return {}
        
        # 确定主要材料类型
        material_types = [info.get('material_type', 'UNKNOWN') for info in material_infos]
        from collections import Counter
        most_common_type = Counter(material_types).most_common(1)[0][0]
        
        # 平均化数值属性
        avg_confidence = np.mean([info.get('material_confidence', 0.0) for info in material_infos])
        avg_cohesion = np.mean([info.get('cohesion', 0.0) for info in material_infos])
        avg_color_var = np.mean([info.get('color_variance', 0.0) for info in material_infos])
        
        # 合并颜色（取频率最高的颜色）
        all_colors = []
        for info in material_infos:
            colors = info.get('dominant_colors', [])
            all_colors.extend(colors)
        
        # 简化：取第一个材料的颜色
        merged_colors = material_infos[0].get('dominant_colors', []) if material_infos else []
        
        return {
            'material_type': most_common_type,
            'material_confidence': float(avg_confidence),
            'cohesion': float(avg_cohesion),
            'color_variance': float(avg_color_var),
            'dominant_colors': merged_colors[:3]  # 保留最多3个颜色
        }

# ==================== 数据结构定义 ====================

@dataclass
class EnhancedDetectionResult:
    """增强检测结果数据类"""
    x: int
    y: int
    width: int
    height: int
    label: str  # 'enemy' 或 'ally' 或 'unknown' 或 'material_type'
    confidence: float
    material_type: str
    material_confidence: float
    cohesion: float
    complexity: float
    has_shadow: bool
    shadow_intensity: float
    light_direction: Dict[str, float]
    perspective_score: float
    estimated_importance: float
    is_natural_scenery: bool
    
    def to_dict(self):
        """转换为字典"""
        return {
            'bbox': {
                'x': self.x,
                'y': self.y,
                'width': self.width,
                'height': self.height
            },
            'label': self.label,
            'confidence': self.confidence,
            'material': {
                'type': self.material_type,
                'confidence': self.material_confidence,
                'cohesion': self.cohesion
            },
            'visual_features': {
                'complexity': self.complexity,
                'has_shadow': self.has_shadow,
                'shadow_intensity': self.shadow_intensity,
                'light_direction': self.light_direction,
                'perspective_score': self.perspective_score
            },
            'importance': self.estimated_importance,
            'is_natural_scenery': self.is_natural_scenery
        }

@dataclass
class EnhancedFrameAnalysisResult:
    """增强帧分析结果"""
    frame_id: int
    timestamp: float
    processing_time: float
    frame_size: Tuple[int, int]
    detections: List[EnhancedDetectionResult]
    macro_optimization_stats: Dict[str, Any]
    light_analysis: Dict[str, Any]
    
    def to_dict(self):
        """转换为字典"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'processing_time': self.processing_time,
            'frame_size': {
                'width': self.frame_size[0],
                'height': self.frame_size[1]
            },
            'detection_count': len(self.detections),
            'detections': [det.to_dict() for det in self.detections],
            'optimization': self.macro_optimization_stats,
            'light_analysis': self.light_analysis
        }
    
    def to_json(self):
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_binary(self):
        """转换为二进制格式"""
        # 扩展二进制格式以包含增强信息
        binary_data = bytearray()
        
        # 头部信息
        binary_data.extend(struct.pack('<I', self.frame_id))  # 帧ID
        binary_data.extend(struct.pack('<d', self.timestamp))  # 时间戳
        binary_data.extend(struct.pack('<H', self.frame_size[0]))  # 宽度
        binary_data.extend(struct.pack('<H', self.frame_size[1]))  # 高度
        
        # 检测数量
        detection_count = min(len(self.detections), 255)
        binary_data.append(detection_count)
        
        # 每个检测
        for i in range(detection_count):
            det = self.detections[i]
            
            # 边界框
            binary_data.extend(struct.pack('<H', det.x))
            binary_data.extend(struct.pack('<H', det.y))
            binary_data.extend(struct.pack('<H', det.width))
            binary_data.extend(struct.pack('<H', det.height))
            
            # 标签编码
            if det.label == 'enemy':
                label_code = 1
            elif det.label == 'ally':
                label_code = 2
            elif det.label == 'unknown':
                label_code = 3
            else:
                label_code = 4  # 材料类型
            
            binary_data.append(label_code)
            
            # 置信度
            binary_data.extend(struct.pack('<f', det.confidence))
            
            # 材料类型编码
            material_code = {
                'UNKNOWN': 0, 'METAL': 1, 'WOOD': 2, 'STONE': 3,
                'PLASTIC': 4, 'TEXTILE': 5, 'GLASS': 6, 'LIQUID': 7,
                'SKIN': 8, 'VEGETATION': 9, 'SKY': 10
            }.get(det.material_type, 0)
            binary_data.append(material_code)
            
            # 材料置信度
            binary_data.extend(struct.pack('<f', det.material_confidence))
            
            # 粘合度
            binary_data.extend(struct.pack('<f', det.cohesion))
            
            # 复杂度
            binary_data.extend(struct.pack('<f', det.complexity))
            
            # 阴影标志
            binary_data.append(1 if det.has_shadow else 0)
            
            # 阴影强度
            binary_data.extend(struct.pack('<f', det.shadow_intensity))
            
            # 重要性
            binary_data.extend(struct.pack('<f', det.estimated_importance))
        
        return bytes(binary_data)

# ==================== 屏幕捕获管理器 - 优化版 ====================

class OptimizedScreenCapture:
    """优化屏幕捕获管理器 - 性能优化版"""
    
    def __init__(self, screen_region=None, monitor_index=0, capture_method='auto'):
        """
        初始化屏幕捕获管理器
        
        Args:
            screen_region: 屏幕区域 (left, top, width, height)
            monitor_index: 显示器索引
            capture_method: 捕获方法 ('pil', 'pyautogui', 'auto')
        """
        self.screen_region = screen_region
        self.monitor_index = monitor_index
        self.capture_method = capture_method
        self.system = platform.system()
        
        # 获取屏幕信息
        self._get_screen_info()
        
        # 性能优化
        self.capture_times = deque(maxlen=100)
        self.capture_failures = 0
        self.max_failures = 10
        
        # 自适应捕获方法选择
        if capture_method == 'auto':
            self.capture_method = self._select_best_capture_method()
        
        print(f"优化屏幕捕获初始化完成 (系统: {self.system}, 方法: {self.capture_method})")
        print(f"屏幕区域: {self.screen_region if self.screen_region else '全屏'}")
    
    def _get_screen_info(self):
        """获取屏幕信息"""
        try:
            monitors = screeninfo.get_monitors()
            if monitors and self.monitor_index < len(monitors):
                monitor = monitors[self.monitor_index]
                self.screen_width = monitor.width
                self.screen_height = monitor.height
                
                # 如果未指定区域，使用全屏
                if not self.screen_region:
                    if self.monitor_index == 0:
                        self.screen_region = (0, 0, self.screen_width, self.screen_height)
                    else:
                        self.screen_region = (monitor.x, monitor.y, self.screen_width, self.screen_height)
                        
                print(f"显示器{self.monitor_index}: {self.screen_width}x{self.screen_height}")
            else:
                self.screen_width, self.screen_height = pyautogui.size()
                print(f"使用pyautogui检测屏幕分辨率: {self.screen_width}x{self.screen_height}")
                
                if not self.screen_region:
                    self.screen_region = (0, 0, self.screen_width, self.screen_height)
        except Exception as e:
            print(f"获取屏幕信息失败: {e}")
            self.screen_width, self.screen_height = 1920, 1080
            if not self.screen_region:
                self.screen_region = (0, 0, self.screen_width, self.screen_height)
    
    def _select_best_capture_method(self) -> str:
        """选择最佳捕获方法"""
        methods = ['pil', 'pyautogui']
        
        # 测试每种方法的速度
        test_results = {}
        
        for method in methods:
            try:
                start_time = time.time()
                
                if method == 'pil':
                    screenshot = ImageGrab.grab()
                    img_array = np.array(screenshot)
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:  # pyautogui
                    screenshot = pyautogui.screenshot()
                    img_array = np.array(screenshot)
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                capture_time = time.time() - start_time
                test_results[method] = capture_time
                
                print(f"方法 {method} 测试: {capture_time*1000:.1f}ms")
                
            except Exception as e:
                print(f"方法 {method} 测试失败: {e}")
                test_results[method] = float('inf')
        
        # 选择最快的方法
        best_method = min(test_results, key=test_results.get)
        print(f"选择捕获方法: {best_method} ({test_results[best_method]*1000:.1f}ms)")
        
        return best_method
    
    def capture(self) -> Optional[np.ndarray]:
        """捕获屏幕"""
        if self.capture_failures >= self.max_failures:
            print(f"截图失败次数过多({self.capture_failures}次)，暂停截图")
            return None
        
        start_time = time.time()
        
        try:
            if self.capture_method == 'pil':
                result = self._capture_with_pil()
            else:
                result = self._capture_with_pyautogui()
            
            if result is None:
                raise ValueError("捕获结果为空")
            
            # 计算捕获时间
            capture_time = time.time() - start_time
            self.capture_times.append(capture_time)
            
            return result
            
        except Exception as e:
            print(f"截图失败: {e}")
            self.capture_failures += 1
            return None
    
    def _capture_with_pil(self) -> np.ndarray:
        """使用PIL捕获屏幕"""
        try:
            if self.screen_region and len(self.screen_region) == 4:
                left, top, width, height = self.screen_region
                left = max(0, left)
                top = max(0, top)
                width = max(1, width)
                height = max(1, height)
                
                screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
            else:
                screenshot = ImageGrab.grab()
            
            img_array = np.array(screenshot)
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
            return img_array
            
        except Exception as e:
            print(f"PIL截图失败: {e}")
            return None
    
    def _capture_with_pyautogui(self) -> np.ndarray:
        """使用pyautogui捕获屏幕"""
        try:
            if self.screen_region and len(self.screen_region) == 4:
                left, top, width, height = self.screen_region
                left = max(0, left)
                top = max(0, top)
                width = max(1, width)
                height = max(1, height)
                
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
            else:
                screenshot = pyautogui.screenshot()
            
            img_array = np.array(screenshot)
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
            self.capture_failures = 0
            return img_array
            
        except Exception as e:
            print(f"pyautogui截图失败: {e}")
            return None
    
    def get_performance_info(self) -> Dict[str, Any]:
        """获取性能信息"""
        if not self.capture_times:
            return {"avg_capture_time": 0, "failures": self.capture_failures}
        
        avg_time = sum(self.capture_times) / len(self.capture_times)
        return {
            "avg_capture_time_ms": avg_time * 1000,
            "min_capture_time_ms": min(self.capture_times) * 1000,
            "max_capture_time_ms": max(self.capture_times) * 1000,
            "failures": self.capture_failures
        }
    
    def reset_failures(self):
        """重置失败计数"""
        self.capture_failures = 0

# ==================== 增强运动目标检测器 ====================

class EnhancedMotionDetector:
    """增强运动目标检测器 - 结合材料分析"""
    
    def __init__(self, 
                 min_area: int = 200, 
                 threshold: int = 25,
                 use_material_filter: bool = True,
                 material_importance_threshold: float = 0.3):
        """
        初始化增强运动检测器
        
        Args:
            min_area: 最小检测面积
            threshold: 运动阈值
            use_material_filter: 是否使用材料过滤器
            material_importance_threshold: 材料重要性阈值
        """
        self.min_area = min_area
        self.threshold = threshold
        self.use_material_filter = use_material_filter
        self.material_threshold = material_importance_threshold
        
        self.prev_gray = None
        self.prev_frame = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 材料分析器
        self.color_analyzer = ColorMaterialAnalyzer()
        self.macro_optimizer = MacroOptimizer()
        
        print(f"增强运动目标检测器初始化完成")
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测运动目标并分析材料"""
        if frame is None or frame.size == 0:
            return []
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 如果是第一帧，初始化
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame.copy()
            return []
        
        # 计算帧差
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        thresh = cv2.dilate(thresh, self.kernel, iterations=1)
        thresh = cv2.erode(thresh, self.kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 更新前一帧
        self.prev_gray = gray
        self.prev_frame = frame.copy()
        
        # 处理检测到的区域
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 15 or h < 15:
                continue
            
            # 提取区域图像
            region_img = frame[y:y+h, x:x+w]
            if region_img.size == 0:
                continue
            
            # 分析材料
            material_info = self.color_analyzer.analyze_color_material(region_img)
            
            # 应用材料过滤器
            if self.use_material_filter:
                importance = self._calculate_region_importance(region_img, material_info)
                if importance < self.material_threshold:
                    continue  # 跳过低重要性区域
            
            # 创建检测结果
            detection = {
                'bbox': (x, y, w, h),
                'area': area,
                'material_info': material_info,
                'importance': self._calculate_region_importance(region_img, material_info),
                'is_moving': True,
                'motion_intensity': np.mean(frame_diff[y:y+h, x:x+w]) / 255.0
            }
            
            detections.append(detection)
        
        # 宏优化处理
        if detections:
            bboxes = [d['bbox'] for d in detections]
            optimized_results = self.macro_optimizer.optimize_processing(frame, bboxes)
            
            # 合并优化结果
            for opt_result in optimized_results:
                if opt_result.get('should_process', True):
                    x, y, w, h = opt_result['bbox']
                    
                    # 查找对应的运动检测
                    motion_match = None
                    for det in detections:
                        det_x, det_y, det_w, det_h = det['bbox']
                        # 检查重叠
                        overlap = self._calculate_overlap(
                            (x, y, w, h), (det_x, det_y, det_w, det_h)
                        )
                        if overlap > 0.3:  # 30%重叠
                            motion_match = det
                            break
                    
                    # 创建增强检测
                    enhanced_det = self._create_enhanced_detection(
                        opt_result, motion_match, frame
                    )
                    
                    if enhanced_det:
                        # 替换或添加
                        if motion_match:
                            idx = detections.index(motion_match)
                            detections[idx] = enhanced_det
                        else:
                            detections.append(enhanced_det)
        
        return detections
    
    def _calculate_region_importance(self, region_img: np.ndarray, material_info: Dict[str, Any]) -> float:
        """计算区域重要性"""
        # 基于材料类型、大小、位置等计算重要性
        importance = 0.0
        
        # 1. 材料类型重要性
        material_type = material_info.get('material_type', 'UNKNOWN')
        type_importance = {
            'SKIN': 0.9, 'METAL': 0.8, 'GLASS': 0.7,
            'TEXTILE': 0.6, 'PLASTIC': 0.5, 'WOOD': 0.4,
            'STONE': 0.3, 'VEGETATION': 0.2, 'SKY': 0.1,
            'LIQUID': 0.6, 'UNKNOWN': 0.5
        }.get(material_type, 0.5)
        
        importance += type_importance * 0.4
        
        # 2. 区域大小重要性
        size_importance = min(region_img.size / 5000.0, 1.0)
        importance += size_importance * 0.3
        
        # 3. 颜色显著性（与背景的对比度）
        if hasattr(self, 'prev_frame') and self.prev_frame is not None:
            bg_region = self.prev_frame[
                max(0, region_img.shape[0]//2 - 10):min(self.prev_frame.shape[0], region_img.shape[0]//2 + 10),
                max(0, region_img.shape[1]//2 - 10):min(self.prev_frame.shape[1], region_img.shape[1]//2 + 10)
            ]
            
            if bg_region.size > 0:
                bg_color = np.mean(bg_region, axis=(0, 1))
                region_color = np.mean(region_img, axis=(0, 1))
                color_diff = np.linalg.norm(bg_color - region_color) / 441.67
                importance += color_diff * 0.3
        
        return min(importance, 1.0)
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的重叠比例"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = w1 * h1
            area2 = w2 * h2
            
            # 返回相对于第一个框的重叠比例
            return inter_area / area1 if area1 > 0 else 0
        
        return 0.0
    
    def _create_enhanced_detection(self, 
                                  opt_result: Dict[str, Any], 
                                  motion_match: Optional[Dict[str, Any]],
                                  frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """创建增强检测结果"""
        bbox = opt_result.get('bbox')
        if not bbox:
            return None
        
        x, y, w, h = bbox
        region_img = frame[y:y+h, x:x+w]
        
        if region_img.size == 0:
            return None
        
        # 分析光影
        light_analyzer = LightShadowAnalyzer()
        light_info = light_analyzer.analyze_light_structure(region_img)
        
        # 获取材料信息
        material_info = opt_result.get('material_info', {})
        
        # 创建检测结果
        detection = {
            'bbox': bbox,
            'area': w * h,
            'material_info': material_info,
            'light_info': light_info,
            'importance': opt_result.get('estimated_importance', 0.5),
            'is_moving': motion_match is not None if motion_match else False,
            'motion_intensity': motion_match.get('motion_intensity', 0.0) if motion_match else 0.0,
            'complexity': opt_result.get('complexity', 0.5),
            'is_natural_scenery': opt_result.get('is_natural_scenery', False),
            'should_process': opt_result.get('should_process', True)
        }
        
        return detection

# ==================== 增强视觉流水线 ====================

class EnhancedVisionPipeline:
    """增强视觉流水线 - 集成材料建模、光影分析和宏观优化"""
    
    def __init__(self, 
                 target_fps: int = 10,
                 screen_region=None,
                 output_format: str = "json",
                 include_screenshot: bool = False,
                 monitor_index: int = 0,
                 use_geda: bool = True):
        """
        初始化增强视觉流水线
        
        Args:
            target_fps: 目标帧率
            screen_region: 屏幕区域
            output_format: 输出格式
            include_screenshot: 是否包含截图Base64
            monitor_index: 显示器索引
            use_geda: 是否使用GEDA算法
        """
        self.target_fps = target_fps
        self.frame_time_target = 1.0 / target_fps
        self.include_screenshot = include_screenshot
        self.monitor_index = monitor_index
        self.use_geda = use_geda
        
        # 初始化组件
        self.screen_capture = OptimizedScreenCapture(screen_region, monitor_index)
        self.motion_detector = EnhancedMotionDetector(
            min_area=150,
            use_material_filter=True,
            material_importance_threshold=0.3
        )
        self.color_analyzer = ColorMaterialAnalyzer()
        self.light_analyzer = LightShadowAnalyzer()
        self.macro_optimizer = MacroOptimizer()
        
        # 结果输出管理器（复用原有类）
        from visual_module_backup import ResultOutputManager  # 假设原有类已定义
        self.output_manager = ResultOutputManager(output_format=output_format)
        
        # GEDA智能体
        self.geda_agent = None
        if use_geda:
            self.geda_agent = GEDA_Vision_Agent(vision_pipeline=self)
        
        # 控制变量
        self.running = False
        self.processing_thread = None
        
        # 统计信息
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        self.optimization_stats = deque(maxlen=100)
        
        print(f"增强视觉流水线初始化完成")
        print(f"目标帧率: {target_fps}FPS, 使用GEDA: {use_geda}")
    
    def process_frame(self, frame: np.ndarray) -> EnhancedFrameAnalysisResult:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            增强帧分析结果
        """
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            return self._create_empty_result()
        
        frame_height, frame_width = frame.shape[:2]
        
        # 1. 运动目标检测与材料分析
        raw_detections = self.motion_detector.detect(frame)
        
        # 2. 全局光影分析
        global_light_analysis = self.light_analyzer.analyze_light_structure(frame)
        
        # 3. 转换检测结果为标准格式
        enhanced_detections = []
        for raw_det in raw_detections:
            enhanced_det = self._convert_to_enhanced_detection(raw_det)
            if enhanced_det:
                enhanced_detections.append(enhanced_det)
        
        # 4. 如果需要，应用GEDA分析
        if self.use_geda and self.geda_agent and enhanced_detections:
            # 计算环境压力
            environment_pressure = self.geda_agent.analyze_environment(
                self._create_geda_compatible_result(enhanced_detections)
            )
            
            # 建模未知物体
            unknown_models = self.geda_agent.model_unknown_objects(
                self._convert_to_detection_results(enhanced_detections)
            )
            
            # 如果压力高，生成决策
            if environment_pressure > 0.5:
                gene_sequence = self.geda_agent.generate_gene_sequence()
                action_plan = self.geda_agent.express_decision(gene_sequence)
                
                # 可以在此处执行决策
                # self.geda_agent.execute_decision(action_plan, simulate=True)
        
        # 5. 获取优化统计
        macro_stats = self.macro_optimizer.get_statistics()
        self.optimization_stats.append(macro_stats)
        
        # 6. 计算处理时间
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.detection_counts.append(len(enhanced_detections))
        
        # 7. 创建结果
        result = EnhancedFrameAnalysisResult(
            frame_id=self.frame_count,
            timestamp=time.time(),
            processing_time=processing_time,
            frame_size=(frame_width, frame_height),
            detections=enhanced_detections,
            macro_optimization_stats=macro_stats,
            light_analysis=global_light_analysis
        )
        
        return result
    
    def _convert_to_enhanced_detection(self, raw_detection: Dict[str, Any]) -> Optional[EnhancedDetectionResult]:
        """转换原始检测为增强检测结果"""
        if not raw_detection:
            return None
        
        bbox = raw_detection.get('bbox')
        if not bbox:
            return None
        
        x, y, w, h = bbox
        
        material_info = raw_detection.get('material_info', {})
        light_info = raw_detection.get('light_info', {})
        
        # 确定标签（基于材料类型和运动）
        material_type = material_info.get('material_type', 'UNKNOWN')
        is_moving = raw_detection.get('is_moving', False)
        
        if material_type in ['SKIN', 'TEXTILE'] and is_moving:
            label = 'enemy' if random.random() > 0.5 else 'ally'  # 简化逻辑
        elif material_type in ['METAL', 'GLASS']:
            label = 'object'
        else:
            label = material_type.lower()
        
        return EnhancedDetectionResult(
            x=x,
            y=y,
            width=w,
            height=h,
            label=label,
            confidence=raw_detection.get('importance', 0.5),
            material_type=material_type,
            material_confidence=material_info.get('material_confidence', 0.0),
            cohesion=material_info.get('cohesion', 0.0),
            complexity=raw_detection.get('complexity', 0.5),
            has_shadow=light_info.get('has_shadow', False),
            shadow_intensity=light_info.get('shadow_intensity', 0.0),
            light_direction=light_info.get('light_direction', {'angle_radians': 0.0, 'angle_degrees': 0.0, 'strength': 0.0}),
            perspective_score=material_info.get('perspective_score', 0.0),
            estimated_importance=raw_detection.get('importance', 0.5),
            is_natural_scenery=raw_detection.get('is_natural_scenery', False)
        )
    
    def _create_geda_compatible_result(self, detections: List[EnhancedDetectionResult]):
        """创建GEDA兼容的结果格式"""
        # 这是一个简化版本，实际需要根据GEDA_Vision_Agent的接口调整
        class SimpleResult:
            def __init__(self, detections):
                self.detections = detections
        
        return SimpleResult(detections)
    
    def _convert_to_detection_results(self, enhanced_dets: List[EnhancedDetectionResult]):
        """转换为DetectionResult列表（兼容原有GEDA接口）"""
        from visual_module_backup import DetectionResult  # 假设原有类已定义
        
        results = []
        for det in enhanced_dets:
            results.append(DetectionResult(
                x=det.x,
                y=det.y,
                width=det.width,
                height=det.height,
                label=det.label,
                confidence=det.confidence
            ))
        return results
    
    def _create_empty_result(self) -> EnhancedFrameAnalysisResult:
        """创建空结果"""
        return EnhancedFrameAnalysisResult(
            frame_id=self.frame_count,
            timestamp=time.time(),
            processing_time=0.0,
            frame_size=(0, 0),
            detections=[],
            macro_optimization_stats={},
            light_analysis={}
        )
    
    def run_processing_loop(self):
        """主处理循环"""
        print("增强视觉流水线启动...")
        print("按 Ctrl+C 停止程序")
        
        self.running = True
        self.frame_count = 0
        consecutive_failures = 0
        
        last_stat_time = time.time()
        last_frame_time = time.time()
        
        # 主循环
        while self.running:
            try:
                frame_start_time = time.time()
                
                # 1. 捕获屏幕
                frame = self.screen_capture.capture()
                
                if frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures > 5:
                        print("连续截图失败5次，暂停处理")
                        time.sleep(1.0)
                        self.screen_capture.reset_failures()
                        consecutive_failures = 0
                    continue
                
                consecutive_failures = 0
                
                # 2. 处理帧
                result = self.process_frame(frame)
                
                # 3. 更新统计信息
                self.frame_count += 1
                
                # 4. 输出结果
                self.output_manager.add_result(result)
                
                # 5. 控制帧率
                elapsed_time = time.time() - frame_start_time
                sleep_time = max(0, self.frame_time_target - elapsed_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 6. 定期显示统计信息
                current_time = time.time()
                if current_time - last_stat_time >= 5.0:
                    self._print_statistics()
                    last_stat_time = current_time
                    
            except KeyboardInterrupt:
                print("\n收到中断信号，停止处理...")
                break
            except Exception as e:
                print(f"处理帧时出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
        
        self.stop()
    
    def _print_statistics(self):
        """打印统计信息"""
        if not self.processing_times:
            return
        
        # 计算统计数据
        avg_process_time = sum(self.processing_times) / len(self.processing_times)
        max_process_time = max(self.processing_times)
        
        avg_detections = sum(self.detection_counts) / len(self.detection_counts) if self.detection_counts else 0
        
        # 截图性能
        capture_perf = self.screen_capture.get_performance_info()
        
        # 优化统计
        if self.optimization_stats:
            latest_opt = self.optimization_stats[-1]
            skipped_ratio = latest_opt.get('skipped_ratio', 0.0) * 100
        else:
            skipped_ratio = 0.0
        
        # 输出统计
        stats_msg = (
            f"帧数: {self.frame_count:5d} | "
            f"处理: {avg_process_time*1000:5.1f}ms | "
            f"截图: {capture_perf['avg_capture_time_ms']:5.1f}ms | "
            f"检测: {avg_detections:4.1f}/帧 | "
            f"跳过: {skipped_ratio:4.1f}% | "
            f"队列: {self.output_manager.get_statistics()['queue_size']:2d}"
        )
        
        print(stats_msg)
        
        # 检查性能问题
        if max_process_time * 1000 > 200:  # 超过200ms
            print(f"警告: 最大处理时间 {max_process_time*1000:.1f}ms 过高!")
    
    def start(self):
        """启动处理流水线"""
        if self.running:
            return
        
        # 在新线程中运行处理循环
        self.processing_thread = threading.Thread(
            target=self.run_processing_loop, 
            daemon=True,
            name="EnhancedVisionPipeline"
        )
        self.processing_thread.start()
        
        print("增强视觉流水线已启动")
    
    def stop(self):
        """停止处理流水线"""
        self.running = False
        
        # 停止所有组件
        print("正在停止所有组件...")
        self.output_manager.stop()
        
        if self.geda_agent:
            self.geda_agent.save_state()
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
        
        # 打印最终统计
        print("\n" + "="*60)
        print("程序停止 - 增强视觉流水线最终统计")
        print("="*60)
        
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            print(f"总帧数: {self.frame_count}")
            print(f"平均处理时间: {avg_time*1000:.1f}ms")
        
        if self.optimization_stats and len(self.optimization_stats) > 0:
            latest_stats = self.optimization_stats[-1]
            print(f"处理区域数: {latest_stats.get('processed_regions', 0)}")
            print(f"跳过区域数: {latest_stats.get('skipped_regions', 0)}")
            print(f"跳过比例: {latest_stats.get('skipped_ratio', 0)*100:.1f}%")
        
        output_stats = self.output_manager.get_statistics()
        print(f"总输出结果: {output_stats['total_outputs']}")
        print("="*60)
    
    def run(self):
        """运行流水线（阻塞版本）"""
        self.start()
        
        try:
            # 等待处理线程结束
            if self.processing_thread:
                self.processing_thread.join()
        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止...")
            self.stop()

# ==================== 主程序 ====================

def setup_environment():
    """设置运行环境"""
    print("=" * 60)
    print("增强版视觉模块 - 材料建模与光影分析系统")
    print("=" * 60)
    
    system = platform.system()
    print(f"检测到系统: {system}")
    
    if system == "Windows":
        print("执行Windows系统优化设置...")
        
        if not is_admin():
            print("程序需要管理员权限以进行屏幕捕获...")
            if request_admin_privileges():
                return False
    
    import_required_libraries()
    
    return True

def parse_arguments():
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版视觉模块')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'geda'], default='enhanced', 
                       help='运行模式: basic(基础), enhanced(增强), geda(GEDA增强)')
    parser.add_argument('--fps', type=int, default=10, help='目标帧率 (默认: 10)')
    parser.add_argument('--region', type=str, help='屏幕区域: left,top,width,height')
    parser.add_argument('--format', choices=['json', 'binary', 'both'], default='json', help='输出格式')
    parser.add_argument('--screenshot', action='store_true', help='包含截图Base64')
    parser.add_argument('--monitor', type=int, default=0, help='显示器索引 (默认: 0)')
    parser.add_argument('--cohesion-threshold', type=float, default=0.7, help='粘合度阈值 (默认: 0.7)')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    return parser.parse_args()

def main():
    """主函数"""
    if not setup_environment():
        return
    
    # 解析命令行参数
    args = parse_arguments()
    
    print("\n配置信息:")
    print(f"  运行模式: {args.mode}")
    print(f"  目标帧率: {args.fps} FPS")
    print(f"  输出格式: {args.format}")
    print(f"  包含截图: {args.screenshot}")
    print(f"  显示器索引: {args.monitor}")
    print(f"  粘合度阈值: {args.cohesion_threshold}")
    print("-" * 60)
    
    # 解析屏幕区域
    screen_region = None
    if args.region:
        try:
            screen_region = tuple(map(int, args.region.split(',')))
            if len(screen_region) != 4:
                raise ValueError
        except:
            print(f"无效的屏幕区域格式: {args.region}")
            print("请使用格式: left,top,width,height")
            return
    
    try:
        if args.mode == 'basic':
            # 使用原有基础流水线
            from visual_module_backup import BackgroundVisionPipeline
            pipeline = BackgroundVisionPipeline(
                target_fps=args.fps,
                screen_region=screen_region,
                output_format=args.format,
                include_screenshot=args.screenshot,
                monitor_index=args.monitor
            )
        elif args.mode == 'enhanced':
            # 使用增强流水线
            pipeline = EnhancedVisionPipeline(
                target_fps=args.fps,
                screen_region=screen_region,
                output_format=args.format,
                include_screenshot=args.screenshot,
                monitor_index=args.monitor,
                use_geda=False
            )
        else:  # geda模式
            # 使用GEDA增强流水线
            pipeline = EnhancedVisionPipeline(
                target_fps=args.fps,
                screen_region=screen_region,
                output_format=args.format,
                include_screenshot=args.screenshot,
                monitor_index=args.monitor,
                use_geda=True
            )
        
        # 运行流水线
        pipeline.run()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序结束")

# ==================== 兼容性导入 ====================

# 导入原有模块以保持兼容性
# 注意：这里假设原有模块在同一目录下，名称为visual_module_backup.py
try:
    from visual_module_backup import (
        BaseType, ThreatLevel, DetectionResult, FrameAnalysisResult,
        BackgroundScreenCapture, BackgroundDataCollector, MotionTargetDetector,
        GameObjectClassifier, ResultOutputManager, BackgroundVisionPipeline,
        GEDA_Vision_Agent, EnhancedVisionPipeline as OriginalEnhancedPipeline,
        DeviceInteractionManager
    )
    
    print("成功导入原有模块以保持兼容性")
except ImportError:
    print("警告: 无法导入原有模块，某些功能可能不可用")
    
    # 定义必要的占位符类
    class GEDA_Vision_Agent:
        def __init__(self, *args, **kwargs):
            pass
        def analyze_environment(self, *args, **kwargs):
            return 0.5
        def model_unknown_objects(self, *args, **kwargs):
            return {}
        def generate_gene_sequence(self, *args, **kwargs):
            return "AGCT"
        def express_decision(self, *args, **kwargs):
            return {}
        def save_state(self, *args, **kwargs):
            pass
    
    class ResultOutputManager:
        def __init__(self, *args, **kwargs):
            self.output_queue = queue.Queue()
            self.running = True
        def add_result(self, *args, **kwargs):
            pass
        def stop(self):
            pass
        def get_statistics(self):
            return {'queue_size': 0, 'total_outputs': 0}

if __name__ == "__main__":
    main()