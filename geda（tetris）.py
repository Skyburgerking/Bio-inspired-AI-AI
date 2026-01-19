import pygame
import numpy as np
import random
import math
from collections import deque
import heapq
import time

# 初始化pygame
pygame.init()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)
GRAY = (100, 100, 100)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
DARK_GREEN = (0, 180, 0)
LIGHT_BLUE = (100, 200, 255)
PINK = (255, 100, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (180, 180, 180)

# 俄罗斯方块颜色
COLORS = [
    (0, 255, 255),    # I - 青色
    (0, 0, 255),      # J - 蓝色
    (255, 165, 0),    # L - 橙色
    (255, 255, 0),    # O - 黄色
    (0, 255, 0),      # S - 绿色
    (128, 0, 128),    # T - 紫色
    (255, 0, 0)       # Z - 红色
]

# 游戏参数
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
VISUAL_WIDTH = 8  # 视觉系统的宽度
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH + 500  # 增加视觉模拟区域
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT + 100
FPS = 30  # 降低帧率，让算法有更多时间思考

# 俄罗斯方块形状
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]],  # L
    [[1, 1], [1, 1]],  # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[0, 1, 0], [1, 1, 1]],  # T
    [[1, 1, 0], [0, 1, 1]]   # Z
]

# GEDA碱基常量 - 完整保留原算法
GENE_BASES = ['A', 'G', 'C', 'T', 'X']  # 激进行动/稳定操作/保守策略/灵活调整/突变探索
GENE_COMPLEMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'X': 'X'}

class GEDAVisualSystem:
    """GEDA视觉模拟系统 - 俄罗斯方块版本（完整保留原视觉系统）"""
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # 视觉层：当前游戏状态
        self.visual_layer = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # 基因表达层：基因激活状态（完整保留原算法4个通道）
        self.gene_expression_layer = np.zeros((grid_height, grid_width, 4), dtype=np.float32)
        # 通道0: A碱基(激进)激活度
        # 通道1: G碱基(稳定)激活度
        # 通道2: C碱基(保守)激活度
        # 通道3: T碱基(灵活)激活度
        
        # X碱基(探索)层
        self.exploration_layer = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # 环境压力层
        self.pressure_layer = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # 基因记忆库：位置->基因表达记录（完整保留原记忆系统）
        self.gene_memory = {}
        
        # 视野记忆
        self.vision_memory = deque(maxlen=50)
        
        # 位置评估器 - 用于评估最佳放置位置
        self.position_evaluator = PositionEvaluator(grid_width, grid_height)
        
        # 初始化基因表达模式
        self._initialize_gene_patterns()
        
        # 基因表达历史
        self.expression_history = deque(maxlen=100)
        
    def _initialize_gene_patterns(self):
        """初始化基因表达模式 - 与原始GEDA算法一致"""
        # 边界区域：高C碱基表达（保守）
        self.gene_expression_layer[0, :, 2] = 0.8  # 上边界
        self.gene_expression_layer[-1, :, 2] = 0.8  # 下边界
        self.gene_expression_layer[:, 0, 2] = 0.8  # 左边界
        self.gene_expression_layer[:, -1, 2] = 0.8  # 右边界
        
        # 中心区域：中等G碱基表达（稳定）
        center_y = self.grid_height // 2
        center_x = self.grid_width // 2
        for y in range(max(0, center_y-3), min(self.grid_height, center_y+4)):
            for x in range(max(0, center_x-3), min(self.grid_width, center_x+4)):
                self.gene_expression_layer[y, x, 1] = 0.5
        
        # 四个角落：A碱基表达（激进）
        self.gene_expression_layer[0, 0, 0] = 0.6  # 左上角
        self.gene_expression_layer[0, self.grid_width-1, 0] = 0.6  # 右上角
        self.gene_expression_layer[self.grid_height-1, 0, 0] = 0.6  # 左下角
        self.gene_expression_layer[self.grid_height-1, self.grid_width-1, 0] = 0.6  # 右下角
    
    def update_visual_input(self, board, current_piece, next_piece, gene_chain):
        """更新视觉输入和基因表达（完整保留原算法逻辑）"""
        # 清空视觉层
        self.visual_layer.fill(0)
        
        # 清空探索层
        self.exploration_layer.fill(0)
        
        # 计算当前状态的特征
        board_height = self._calculate_board_height(board)
        holes = self._count_holes(board)
        complete_lines = self._find_complete_lines(board)
        
        # 提取局部基因特征（基于当前游戏状态和位置）
        current_x = current_piece.x if current_piece else self.grid_width // 2
        current_y = current_piece.y if current_piece else 0
        local_genes = self._extract_local_genes(gene_chain, current_x, current_y)
        
        # 更新基因表达层
        self._update_gene_expression(board, current_piece, next_piece, local_genes)
        
        # 更新视觉层
        self._update_visual_field(board, current_piece)
        
        # 更新环境压力层
        self._update_pressure_layer(board, current_piece)
        
        # 更新X碱基探索层
        self._update_exploration_layer(board, local_genes, current_x, current_y)
        
        # 记录视觉记忆
        self.vision_memory.append({
            'position': (current_x, current_y),
            'board_height': board_height,
            'holes': holes,
            'genes': local_genes,
            'visual': self.visual_layer.copy(),
            'pressure': self.pressure_layer[current_y, current_x] if 0 <= current_y < self.grid_height and 0 <= current_x < self.grid_width else 0.5
        })
        
        # 记录基因表达历史
        if 0 <= current_y < self.grid_height and 0 <= current_x < self.grid_width:
            self.expression_history.append({
                'position': (current_x, current_y),
                'expression': self.gene_expression_layer[current_y, current_x].copy(),
                'genes': local_genes,
                'timestamp': time.time()
            })
        
        return local_genes
    
    def _extract_local_genes(self, gene_chain, x, y):
        """根据位置提取局部基因特征（与原始GEDA算法完全一致）"""
        # 使用位置信息作为种子来提取基因
        seed = (x * 17 + y * 31) % max(len(gene_chain), 1)
        gene_window = 5
        
        local_genes = []
        for i in range(gene_window):
            idx = (seed + i) % len(gene_chain)
            local_genes.append(gene_chain[idx])
        
        return local_genes
    
    def _calculate_board_height(self, board):
        """计算棋盘高度"""
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if board[y][x] != 0:
                    return self.grid_height - y
        return 0
    
    def _count_holes(self, board):
        """计算空洞数量"""
        holes = 0
        for x in range(self.grid_width):
            block_found = False
            for y in range(self.grid_height):
                if board[y][x] != 0:
                    block_found = True
                elif block_found and board[y][x] == 0:
                    holes += 1
        return holes
    
    def _find_complete_lines(self, board):
        """查找完整行"""
        complete_lines = []
        for y in range(self.grid_height):
            if all(board[y][x] != 0 for x in range(self.grid_width)):
                complete_lines.append(y)
        return complete_lines
    
    def _update_gene_expression(self, board, current_piece, next_piece, local_genes):
        """更新基因表达层（完整保留原算法逻辑）"""
        if not current_piece:
            return
            
        head_x, head_y = current_piece.x, current_piece.y
        
        # 计算环境压力值
        pressure = self._calculate_local_pressure(board, current_piece)
        
        # 分析局部基因组成
        gene_counts = {'A': 0, 'G': 0, 'C': 0, 'T': 0, 'X': 0}
        for gene in local_genes:
            if gene in gene_counts:
                gene_counts[gene] += 1
        
        total = len(local_genes)
        if total == 0:
            total = 1
        
        # 根据环境压力选择表达模式（完整保留原算法逻辑）
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # 距离当前方块的距离因子
                dist_to_head = math.sqrt((x - head_x)**2 + (y - head_y)**2)
                distance_factor = max(0, 1 - dist_to_head / (self.grid_width * 0.7))
                
                # 根据环境压力选择表达模式
                if pressure > 0.7:  # 高压环境
                    # 表达A(激进)和G(稳定)基因
                    self.gene_expression_layer[y, x, 0] = gene_counts['A'] / total * pressure * distance_factor
                    self.gene_expression_layer[y, x, 1] = gene_counts['G'] / total * (1 - pressure) * distance_factor
                elif pressure < 0.3:  # 低压环境
                    # 表达C(保守)和T(灵活)基因
                    self.gene_expression_layer[y, x, 2] = gene_counts['C'] / total * (1 - pressure) * distance_factor
                    self.gene_expression_layer[y, x, 3] = gene_counts['T'] / total * pressure * distance_factor
                else:  # 中等压力
                    # 平衡表达
                    self.gene_expression_layer[y, x, 0] = gene_counts['A'] / total * 0.3 * distance_factor
                    self.gene_expression_layer[y, x, 1] = gene_counts['G'] / total * 0.4 * distance_factor
                    self.gene_expression_layer[y, x, 2] = gene_counts['C'] / total * 0.3 * distance_factor
                    self.gene_expression_layer[y, x, 3] = gene_counts['T'] / total * 0.4 * distance_factor
                
                # X基因的探索表达（与压力负相关）
                if 'X' in local_genes:
                    explore_intensity = gene_counts['X'] / total * (1 - pressure) * distance_factor
                    self.exploration_layer[y, x, 0] = int(255 * explore_intensity * 0.7)  # 红色通道表示探索
                    self.exploration_layer[y, x, 2] = int(255 * explore_intensity * 0.5)  # 蓝色通道
        
        # 记忆位置基因表达（完整保留原记忆系统）
        if (head_x, head_y) not in self.gene_memory or time.time() - self.gene_memory.get((head_x, head_y), {}).get('timestamp', 0) > 1.0:
            self.gene_memory[(head_x, head_y)] = {
                'genes': local_genes,
                'pressure': pressure,
                'expression': self.gene_expression_layer[head_y, head_x].copy() if 0 <= head_y < self.grid_height and 0 <= head_x < self.grid_width else np.zeros(4),
                'timestamp': time.time()
            }
    
    def _calculate_local_pressure(self, board, current_piece):
        """计算局部环境压力（完整保留原压力计算逻辑）"""
        if not current_piece:
            return 0.5
            
        head_x, head_y = current_piece.x, current_piece.y
        
        # 1. 空间压力
        obstacles = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = head_x + dx, head_y + dy
            if (nx < 0 or nx >= self.grid_width or 
                ny < 0 or ny >= self.grid_height or 
                (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and board[ny][nx] != 0)):
                obstacles += 1
        
        space_pressure = obstacles / 4
        
        # 2. 棋盘高度压力
        board_height = self._calculate_board_height(board)
        height_pressure = board_height / self.grid_height
        
        # 3. 空洞压力
        holes = self._count_holes(board)
        max_holes = self.grid_width * (self.grid_height // 2)
        hole_pressure = min(holes / max_holes, 1.0) if max_holes > 0 else 0
        
        # 4. 完整行压力（没有完整行时压力大）
        complete_lines = len(self._find_complete_lines(board))
        complete_line_pressure = 1.0 - min(complete_lines / 4, 1.0)
        
        # 综合压力（使用原算法的权重比例）
        total_pressure = 0.25 * space_pressure + 0.25 * height_pressure + 0.25 * hole_pressure + 0.25 * complete_line_pressure
        
        return min(max(total_pressure, 0.0), 1.0)
    
    def _update_visual_field(self, board, current_piece):
        """更新视觉层"""
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # 已有方块
                if board[y][x] != 0:
                    color_idx = board[y][x] - 1
                    if color_idx < len(COLORS):
                        self.visual_layer[y, x] = COLORS[color_idx]
                    else:
                        self.visual_layer[y, x] = [150, 150, 150]
                
                # 空位置
                else:
                    # 根据基因表达着色（完整保留原算法颜色映射）
                    gene_expr = self.gene_expression_layer[y, x]
                    if gene_expr[0] > 0.5:  # A碱基主导（激进）
                        self.visual_layer[y, x] = [255, 100, 0]  # 橙色
                    elif gene_expr[1] > 0.5:  # G碱基主导（稳定）
                        self.visual_layer[y, x] = [100, 255, 100]  # 浅绿
                    elif gene_expr[2] > 0.5:  # C碱基主导（保守）
                        self.visual_layer[y, x] = [100, 100, 255]  # 浅蓝
                    elif gene_expr[3] > 0.5:  # T碱基主导（灵活）
                        self.visual_layer[y, x] = [255, 100, 255]  # 粉色
                    else:
                        gray_value = 50 + int(100 * y / self.grid_height)
                        self.visual_layer[y, x] = [gray_value, gray_value, gray_value]
        
        # 绘制当前方块
        if current_piece:
            shape = current_piece.get_rotated_shape()
            for y in range(len(shape)):
                for x in range(len(shape[0])):
                    if shape[y][x]:
                        px, py = current_piece.x + x, current_piece.y + y
                        if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                            # 混合颜色（使用原算法逻辑）
                            original = self.visual_layer[py, px].astype(np.float32)
                            piece_color = np.array(COLORS[current_piece.type], dtype=np.float32)
                            blend_color = (original * 0.6 + piece_color * 0.4).astype(np.uint8)
                            self.visual_layer[py, px] = blend_color
    
    def _update_pressure_layer(self, board, current_piece):
        """更新环境压力层"""
        # 计算高度图
        height_map = [0] * self.grid_width
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if board[y][x] != 0:
                    height_map[x] = self.grid_height - y
                    break
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # 高度压力
                height = self.grid_height - y
                height_pressure = 0
                if height <= height_map[x]:
                    height_pressure = 0.8
                
                # 空洞压力
                hole_pressure = 0
                if y > 0 and board[y][x] == 0 and board[y-1][x] != 0:
                    hole_pressure = 0.6
                
                # 边缘压力
                edge_pressure = 0
                if x == 0 or x == self.grid_width - 1:
                    edge_pressure = 0.3
                
                # 综合压力
                self.pressure_layer[y, x] = 0.5 * height_pressure + 0.3 * hole_pressure + 0.2 * edge_pressure
    
    def _update_exploration_layer(self, board, local_genes, center_x, center_y):
        """更新探索层（X碱基表达）"""
        # 检查是否有X碱基
        if 'X' in local_genes:
            x_count = local_genes.count('X')
            explore_intensity = x_count / len(local_genes) if len(local_genes) > 0 else 0
            
            # 在视野范围内标记探索区域
            explore_radius = int(self.grid_width * 0.4 * explore_intensity)
            
            for dy in range(-explore_radius, explore_radius+1):
                for dx in range(-explore_radius, explore_radius+1):
                    x, y = center_x + dx, center_y + dy
                    if (0 <= x < self.grid_width and 0 <= y < self.grid_height and
                        dx*dx + dy*dy <= explore_radius*explore_radius):
                        
                        # 根据距离设置探索强度
                        dist = math.sqrt(dx*dx + dy*dy)
                        intensity = max(0, 1 - dist / explore_radius) if explore_radius > 0 else 0
                        
                        # 探索层使用紫色表示探索区域
                        self.exploration_layer[y, x, 0] = int(255 * intensity * 0.7)  # R
                        self.exploration_layer[y, x, 2] = int(255 * intensity * 0.9)  # B
    
    def get_gene_activation_vector(self, position):
        """获取位置上的基因激活向量"""
        x, y = position
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.gene_expression_layer[y, x].copy()
        return np.zeros(4)
    
    def get_best_position(self, board, piece):
        """获取最佳放置位置（使用增强的位置评估）"""
        return self.position_evaluator.evaluate_best_position(board, piece)
    
    def render_gene_expression_map(self, surface, x_offset, y_offset, scale):
        """渲染基因表达地图（完整保留原算法渲染逻辑）"""
        # 创建临时表面
        map_surface = pygame.Surface((VISUAL_WIDTH * scale, self.grid_height * scale))
        
        # 绘制基因表达地图（只显示左半部分）
        for y in range(self.grid_height):
            for x in range(VISUAL_WIDTH):
                # 获取基因表达值
                a_expr = self.gene_expression_layer[y, x, 0]  # A碱基（激进）
                g_expr = self.gene_expression_layer[y, x, 1]  # G碱基（稳定）
                c_expr = self.gene_expression_layer[y, x, 2]  # C碱基（保守）
                t_expr = self.gene_expression_layer[y, x, 3]  # T碱基（灵活）
                
                # 确定主导基因
                expressions = [('A', a_expr), ('G', g_expr), ('C', c_expr), ('T', t_expr)]
                expressions.sort(key=lambda x: x[1], reverse=True)
                
                dominant_gene, dominant_value = expressions[0]
                
                # 根据主导基因选择颜色
                if dominant_gene == 'A':  # 激进 - 红色系
                    color = (int(255 * dominant_value), int(100 * dominant_value), 0)
                elif dominant_gene == 'G':  # 稳定 - 绿色系
                    color = (0, int(255 * dominant_value), int(100 * dominant_value))
                elif dominant_gene == 'C':  # 保守 - 蓝色系
                    color = (0, int(100 * dominant_value), int(255 * dominant_value))
                elif dominant_gene == 'T':  # 灵活 - 紫色系
                    color = (int(200 * dominant_value), 0, int(255 * dominant_value))
                else:
                    color = (100, 100, 100)
                
                # 绘制基因表达格子
                rect = pygame.Rect(x * scale, y * scale, scale, scale)
                pygame.draw.rect(map_surface, color, rect)
                
                # 绘制网格线
                pygame.draw.rect(map_surface, (30, 30, 30), rect, 1)
                
                # 如果探索层有值，添加探索标记
                if np.any(self.exploration_layer[y, x] > 0):
                    explore_color = tuple(self.exploration_layer[y, x])
                    pygame.draw.circle(map_surface, explore_color,
                                     (x * scale + scale//2, y * scale + scale//2),
                                     scale//3)
        
        # 绘制到主表面
        surface.blit(map_surface, (x_offset, y_offset))
        
        # 绘制标题
        title_font = pygame.font.SysFont(None, 20)
        title_text = title_font.render("基因表达地图 (A:红/G:绿/C:蓝/T:紫)", True, CYAN)
        surface.blit(title_text, (x_offset + 10, y_offset - 25))
        
        # 绘制图例
        legend_y = y_offset + self.grid_height * scale + 10
        legend_items = [
            ("A:激进行动", (255, 100, 0)),
            ("G:稳定操作", (100, 255, 100)),
            ("C:保守策略", (100, 100, 255)),
            ("T:灵活调整", (200, 0, 200)),
            ("X:突变探索", (255, 0, 255))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            # 颜色方块
            pygame.draw.rect(surface, color, (x_offset + i * 80, legend_y, 15, 15))
            # 文本
            legend_text = title_font.render(text, True, WHITE)
            surface.blit(legend_text, (x_offset + i * 80 + 20, legend_y))
    
    def render_pressure_map(self, surface, x_offset, y_offset, scale):
        """渲染环境压力地图（完整保留原算法渲染逻辑）"""
        # 创建临时表面
        map_surface = pygame.Surface((VISUAL_WIDTH * scale, self.grid_height * scale))
        
        # 绘制压力地图（只显示左半部分）
        for y in range(self.grid_height):
            for x in range(VISUAL_WIDTH):
                pressure = self.pressure_layer[y, x]
                
                # 根据压力值选择颜色（低压力=蓝，高压力=红）
                if pressure < 0.3:
                    # 低压力：蓝色
                    blue_intensity = int(255 * (1 - pressure/0.3))
                    color = (50, 50, blue_intensity)
                elif pressure < 0.7:
                    # 中压力：黄色
                    yellow_intensity = int(255 * (pressure - 0.3) / 0.4)
                    color = (yellow_intensity, yellow_intensity, 50)
                else:
                    # 高压力：红色
                    red_intensity = int(255 * min(1.0, (pressure - 0.7) / 0.3))
                    color = (red_intensity, 50, 50)
                
                # 绘制压力格子
                rect = pygame.Rect(x * scale, y * scale, scale, scale)
                pygame.draw.rect(map_surface, color, rect)
                pygame.draw.rect(map_surface, (30, 30, 30), rect, 1)
        
        # 绘制到主表面
        surface.blit(map_surface, (x_offset, y_offset))
        
        # 绘制标题
        title_font = pygame.font.SysFont(None, 20)
        title_text = title_font.render("环境压力地图 (蓝=低压,红=高压)", True, CYAN)
        surface.blit(title_text, (x_offset + 10, y_offset - 25))

class PositionEvaluator:
    """位置评估器 - 俄罗斯方块版本（增强版）"""
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        
    def evaluate_position(self, board, piece, x, y, rotation):
        """评估特定位置的好坏（多维度评估）"""
        # 创建测试棋盘
        test_board = [row[:] for row in board]
        
        # 获取旋转后的形状
        rotated_shape = self.rotate_shape(piece.shape, rotation)
        
        # 检查位置是否有效
        if not self.is_valid_position(test_board, rotated_shape, x, y):
            return float('-inf')
        
        # 放置方块
        self.place_piece(test_board, rotated_shape, x, y, piece.type + 1)
        
        # 1. 完整行数（最高优先级）
        lines_cleared = 0
        for row_y in range(self.grid_height):
            if all(test_board[row_y][col] != 0 for col in range(self.grid_width)):
                lines_cleared += 1
        
        # 2. 高度惩罚
        max_height = 0
        column_heights = []
        for col in range(self.grid_width):
            height = 0
            for row in range(self.grid_height):
                if test_board[row][col] != 0:
                    height = self.grid_height - row
                    break
            column_heights.append(height)
            if height > max_height:
                max_height = height
        
        # 3. 空洞惩罚
        holes = 0
        for col in range(self.grid_width):
            block_found = False
            for row in range(self.grid_height):
                if test_board[row][col] != 0:
                    block_found = True
                elif block_found and test_board[row][col] == 0:
                    holes += 1
        
        # 4. 凸起惩罚
        bumpiness = 0
        for i in range(self.grid_width - 1):
            bumpiness += abs(column_heights[i] - column_heights[i+1])
        
        # 5. 井深度奖励（为I方块留空间）
        well_depth = self._calculate_well_depth(test_board)
        
        # 6. 行变换惩罚（防止高低不平）
        row_transitions = self._calculate_row_transitions(test_board)
        
        # 综合评分
        score = 0
        score += lines_cleared * 400  # 消除行奖励
        score -= max_height * 10      # 高度惩罚
        score -= holes * 50           # 空洞惩罚
        score -= bumpiness * 5        # 凸起惩罚
        score += well_depth * 10      # 井深度奖励
        score -= row_transitions * 2  # 行变换惩罚
        
        # 7. 战略位置奖励（中间位置更好）
        center_x = self.grid_width // 2
        distance_from_center = abs((x + len(rotated_shape[0]) / 2) - center_x)
        score -= distance_from_center * 3
        
        return score
    
    def _calculate_well_depth(self, board):
        """计算井深度"""
        well_depth = 0
        for col in range(1, self.grid_width - 1):
            left_height = 0
            right_height = 0
            center_height = 0
            
            for row in range(self.grid_height):
                if board[row][col-1] != 0:
                    left_height = max(left_height, self.grid_height - row)
                if board[row][col] != 0:
                    center_height = max(center_height, self.grid_height - row)
                if board[row][col+1] != 0:
                    right_height = max(right_height, self.grid_height - row)
            
            if center_height < left_height and center_height < right_height:
                well_depth += min(left_height, right_height) - center_height
        
        return well_depth
    
    def _calculate_row_transitions(self, board):
        """计算行变换次数"""
        transitions = 0
        for row in range(self.grid_height):
            for col in range(self.grid_width - 1):
                if (board[row][col] == 0) != (board[row][col+1] == 0):
                    transitions += 1
        return transitions
    
    def evaluate_best_position(self, board, piece):
        """评估最佳放置位置"""
        best_score = float('-inf')
        best_position = None
        best_rotation = 0
        
        # 尝试所有可能的旋转
        for rotation in range(4):
            rotated_shape = self.rotate_shape(piece.shape, rotation)
            shape_width = len(rotated_shape[0])
            
            # 尝试所有可能的水平位置
            for x in range(self.grid_width - shape_width + 1):
                # 找到最低的有效Y位置
                y = 0
                while self.is_valid_position(board, rotated_shape, x, y + 1):
                    y += 1
                
                # 评估这个位置
                score = self.evaluate_position(board, piece, x, y, rotation)
                
                if score > best_score:
                    best_score = score
                    best_position = (x, y)
                    best_rotation = rotation
        
        return best_position, best_rotation, best_score
    
    def rotate_shape(self, shape, rotation):
        """旋转形状"""
        rotated = shape
        for _ in range(rotation):
            # 转置并反转每一行（顺时针旋转90度）
            rotated = [[rotated[y][x] for y in range(len(rotated)-1, -1, -1)] 
                      for x in range(len(rotated[0]))]
        return rotated
    
    def is_valid_position(self, board, shape, x, y):
        """检查位置是否有效"""
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col]:
                    board_x = x + col
                    board_y = y + row
                    
                    if (board_x < 0 or board_x >= self.grid_width or
                        board_y < 0 or board_y >= self.grid_height or
                        board[board_y][board_x] != 0):
                        return False
        return True
    
    def place_piece(self, board, shape, x, y, color):
        """放置方块到棋盘"""
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col]:
                    board_y = y + row
                    board_x = x + col
                    if 0 <= board_y < self.grid_height and 0 <= board_x < self.grid_width:
                        board[board_y][board_x] = color

class Tetromino:
    """俄罗斯方块单个方块"""
    def __init__(self, type_idx):
        self.type = type_idx
        self.shape = SHAPES[type_idx]
        self.color = COLORS[type_idx]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        self.rotation = 0
    
    def rotate(self):
        """旋转方块"""
        self.rotation = (self.rotation + 1) % 4
    
    def get_rotated_shape(self):
        """获取旋转后的形状"""
        shape = self.shape
        for _ in range(self.rotation):
            shape = [[shape[y][x] for y in range(len(shape)-1, -1, -1)] 
                    for x in range(len(shape[0]))]
        return shape

class GEDATetris:
    """GEDA视觉增强版俄罗斯方块 - 完全自动运行，完整算法"""
    def __init__(self, inherited_genes=None):
        # 游戏状态
        self.board = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        self.pieces_placed = 0
        
        # 游戏控制
        self.fall_speed = 0.5  # 方块下落的速度（秒）
        self.fall_time = 0
        self.last_fall_time = time.time()
        self.auto_move_delay = 0.1  # 自动移动延迟
        self.last_move_time = time.time()
        
        # GEDA 基因系统（完整保留原算法）
        if inherited_genes:
            self.gene_chain = inherited_genes.copy()
            # 对继承的基因进行小幅变异
            if random.random() < 0.2:
                self.mutate_gene_chain('replace')
        else:
            self.gene_chain = self.initialize_gene_chain()
            
        self.memory_dbs = {}  # 分段记忆数据库（完整保留）
        self.expression_cache = {}
        self.environment_history = deque(maxlen=100)
        
        # 视觉系统（完整保留）
        self.visual_system = GEDAVisualSystem(GRID_WIDTH, GRID_HEIGHT)
        
        # 性能统计
        self.decisions_made = 0
        self.successful_decisions = 0
        self.gene_expressions = 0
        self.mutations_count = 0
        
        # 决策状态
        self.target_position = None
        self.target_rotation = 0
        self.decision_mode = "GEDA"
        self.last_decision_time = 0
        
        # 自动移动状态
        self.move_direction = 0  # -1:左, 0:无, 1:右
        self.rotate_needed = False
        self.hard_drop = False
        self.move_complete = False
        
        # 学习状态（增强）
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.learning_rate = 0.1  # 学习率
        self.exploration_rate = 0.1  # 探索率
        
        # 决策质量追踪
        self.decision_quality_history = deque(maxlen=20)
        self.recent_scores = deque(maxlen=10)
        
        print(f"GEDA Tetris 初始化 - 基因长度: {len(self.gene_chain)}")
    
    def new_piece(self):
        """生成新的方块"""
        return Tetromino(random.randint(0, len(SHAPES) - 1))
    
    def initialize_gene_chain(self):
        """初始化基础基因链（完整保留原算法）"""
        gene_length = 60
        gene_chain = []
        
        # 不同基因段有不同特性（与原贪吃蛇算法一致）
        segments = [
            (12, ['A', 'A', 'G', 'T', 'C', 'A', 'G', 'X', 'T', 'A', 'G', 'C']),
            (15, ['G', 'C', 'T', 'G', 'C', 'G', 'T', 'G', 'C', 'T', 'G', 'C', 'T', 'G', 'C']),
            (12, ['C', 'C', 'T', 'A', 'C', 'C', 'T', 'A', 'C', 'C', 'T', 'A']),
            (10, ['T', 'A', 'G', 'C', 'T', 'A', 'G', 'C', 'T', 'A']),
            (11, ['X', 'A', 'X', 'G', 'X', 'C', 'X', 'T', 'X', 'A', 'X'])
        ]
        
        for segment_length, segment_pattern in segments:
            for i in range(segment_length):
                gene_chain.append(segment_pattern[i % len(segment_pattern)])
        
        return gene_chain[:gene_length]
    
    def calculate_environment_pressure(self):
        """计算环境压力值（完整保留原算法逻辑）"""
        if not self.current_piece:
            return 0.5
            
        head_x, head_y = self.current_piece.x, self.current_piece.y
        
        # 使用视觉系统的压力层
        visual_pressure = 0
        if 0 <= head_y < GRID_HEIGHT and 0 <= head_x < GRID_WIDTH:
            visual_pressure = self.visual_system.pressure_layer[head_y, head_x]
        
        # 原始压力计算（与原算法一致）
        # 1. 空间压力
        obstacles = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = head_x + dx, head_y + dy
            if (nx < 0 or nx >= GRID_WIDTH or 
                ny < 0 or ny >= GRID_HEIGHT or 
                (0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH and self.board[ny][nx] != 0)):
                obstacles += 1
        space_pressure = obstacles / 4
        
        # 2. 棋盘高度压力
        board_height = self.visual_system._calculate_board_height(self.board)
        height_pressure = board_height / GRID_HEIGHT
        
        # 3. 空洞压力
        holes = self.visual_system._count_holes(self.board)
        max_holes = GRID_WIDTH * (GRID_HEIGHT // 2)
        hole_pressure = min(holes / max_holes, 1.0) if max_holes > 0 else 0
        
        # 4. 基因多样性压力
        gene_diversity = len(set(self.gene_chain[:20])) / 5
        diversity_pressure = 1.0 - gene_diversity
        
        # 综合压力（与原算法权重一致）
        total_pressure = (0.2 * space_pressure + 
                         0.25 * height_pressure + 
                         0.25 * hole_pressure + 
                         0.15 * diversity_pressure +
                         0.15 * visual_pressure)
        
        return min(max(total_pressure, 0.0), 1.0)
    
    def extract_environment_features(self):
        """提取环境特征向量（完整保留原算法特征）"""
        features = []
        
        # 8个方向的距离特征（与原算法一致）
        if self.current_piece:
            head_x, head_y = self.current_piece.x, self.current_piece.y
            for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), 
                          (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
                distance = 0
                x, y = head_x, head_y
                while (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and 
                       (x == head_x and y == head_y) or (0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH and self.board[y][x] == 0)):
                    x += dx
                    y += dy
                    distance += 1
                    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                        break
                features.append(min(distance, 10) / 10)
        else:
            features.extend([0.5] * 8)
        
        # 基因表达特征
        if self.current_piece:
            gene_expr = self.visual_system.get_gene_activation_vector(
                (self.current_piece.x, self.current_piece.y)
            )
            features.extend(gene_expr.tolist())
        else:
            features.extend([0] * 4)
        
        # 棋盘特征
        board_height = self.visual_system._calculate_board_height(self.board)
        holes = self.visual_system._count_holes(self.board)
        
        features.append(board_height / GRID_HEIGHT)
        features.append(min(holes / 20, 1.0))
        
        # 方块特征
        if self.current_piece:
            features.append(self.current_piece.type / 7)
        else:
            features.append(0.5)
        
        # 成功率特征
        success_rate = self.successful_decisions / max(self.decisions_made, 1)
        features.append(success_rate)
        
        # 环境压力特征
        features.append(self.calculate_environment_pressure())
        
        return features
    
    def activate_gene_segments(self, features):
        """激活匹配的基因段（完整保留原算法逻辑）"""
        pressure = self.calculate_environment_pressure()
        
        # 根据压力选择基因段（与原算法完全一致）
        if pressure > 0.7:  # 高压环境
            # 激活激进和稳定基因段
            start_idx = 0
            end_idx = len(self.gene_chain) // 2
            segment_type = "aggressive"
        elif pressure < 0.3:  # 低压环境
            # 激活保守和灵活基因段
            start_idx = len(self.gene_chain) // 2
            end_idx = len(self.gene_chain)
            segment_type = "conservative"
        else:  # 中等压力
            # 激活中间基因段
            mid_idx = len(self.gene_chain) // 2
            start_idx = max(0, mid_idx - 8)
            end_idx = min(len(self.gene_chain), mid_idx + 8)
            segment_type = "balanced"
        
        # 提取基因段
        gene_segment = self.gene_chain[start_idx:end_idx]
        
        # 记录激活
        self.gene_expressions += 1
        
        return gene_segment, segment_type
    
    def map_gene_to_action(self, gene_segment, features):
        """将基因序列映射为动作（完整保留原算法映射逻辑）"""
        # 分析基因组成
        gene_counts = {'A': 0, 'G': 0, 'C': 0, 'T': 0, 'X': 0}
        for gene in gene_segment[:15]:
            if gene in gene_counts:
                gene_counts[gene] += 1
        
        total = sum(gene_counts.values())
        if total == 0:
            total = 1
        
        # 基因组成比例
        aggressive_ratio = (gene_counts['A'] + gene_counts['G']) / total
        conservative_ratio = (gene_counts['C'] + gene_counts['T']) / total
        explore_ratio = gene_counts['X'] / total
        
        # 获取A*路径推荐（使用位置评估器）
        best_pos, best_rot, best_score = self.visual_system.get_best_position(
            self.board, self.current_piece
        )
        
        # 保存目标位置
        self.target_position = best_pos
        self.target_rotation = best_rot
        
        # 激进基因主导时：积极寻找最佳位置
        if aggressive_ratio > conservative_ratio and aggressive_ratio > 0.4:
            self.decision_mode = "激进模式"
            if best_pos:
                # 计算移动方向
                if self.current_piece.x < best_pos[0]:
                    self.move_direction = 1  # 右
                elif self.current_piece.x > best_pos[0]:
                    self.move_direction = -1  # 左
                else:
                    self.move_direction = 0
                
                # 检查是否需要旋转
                self.rotate_needed = best_rot != self.current_piece.rotation
                
                # 激进模式可能使用硬降
                pressure = self.calculate_environment_pressure()
                self.hard_drop = random.random() < (0.3 + pressure * 0.4)
            else:
                # 没有最佳位置，保守移动
                self.decision_mode = "保守模式"
                self.move_direction = 0
                self.rotate_needed = False
                self.hard_drop = False
        
        # 探索基因活跃时：随机探索
        elif explore_ratio > 0.2:
            self.decision_mode = "探索模式"
            # X碱基带来随机探索
            if random.random() < self.exploration_rate:
                # 随机探索
                if random.random() < 0.3:
                    self.move_direction = random.choice([-1, 0, 1])
                else:
                    self.move_direction = 0
                
                self.rotate_needed = random.random() < 0.3
                self.hard_drop = random.random() < 0.1
            else:
                # 使用最佳位置
                if best_pos:
                    if self.current_piece.x < best_pos[0]:
                        self.move_direction = 1
                    elif self.current_piece.x > best_pos[0]:
                        self.move_direction = -1
                    else:
                        self.move_direction = 0
                    self.rotate_needed = best_rot != self.current_piece.rotation
                    self.hard_drop = False
        
        # 保守基因主导时：安全第一
        else:
            self.decision_mode = "保守模式"
            if best_pos:
                # 缓慢移动到最佳位置
                if self.current_piece.x < best_pos[0]:
                    self.move_direction = 1
                elif self.current_piece.x > best_pos[0]:
                    self.move_direction = -1
                else:
                    self.move_direction = 0
                self.rotate_needed = best_rot != self.current_piece.rotation
                # 保守模式很少硬降
                self.hard_drop = random.random() < 0.05
            else:
                self.move_direction = 0
                self.rotate_needed = False
                self.hard_drop = False
        
        return self.decision_mode
    
    def mutate_gene_chain(self, mutation_type=None):
        """基因变异（完整保留原算法5种变异类型）"""
        if mutation_type is None:
            mutation_type = random.choice(['insert', 'replace', 'delete', 'reorder', 'complement'])
        
        self.mutations_count += 1
        
        if mutation_type == 'insert' and len(self.gene_chain) < 100:
            # 插入突变段
            mutation_length = random.randint(1, 3)
            mutation_genes = [random.choice(GENE_BASES) for _ in range(mutation_length)]
            insert_pos = random.randint(0, len(self.gene_chain))
            self.gene_chain = (self.gene_chain[:insert_pos] + 
                              mutation_genes + 
                              self.gene_chain[insert_pos:])
            
        elif mutation_type == 'replace':
            # 替换碱基
            replace_pos = random.randint(0, len(self.gene_chain)-1)
            self.gene_chain[replace_pos] = random.choice(GENE_BASES)
            
        elif mutation_type == 'delete' and len(self.gene_chain) > 20:
            # 删除碱基
            delete_pos = random.randint(0, len(self.gene_chain)-1)
            self.gene_chain.pop(delete_pos)
            
        elif mutation_type == 'reorder' and len(self.gene_chain) > 10:
            # 重新排序基因段
            start = random.randint(0, len(self.gene_chain)-5)
            end = random.randint(start+2, min(start+5, len(self.gene_chain)))
            segment = self.gene_chain[start:end]
            random.shuffle(segment)
            self.gene_chain[start:end] = segment
            
        elif mutation_type == 'complement':
            # 互补配对变异
            mutate_pos = random.randint(0, len(self.gene_chain)-1)
            original = self.gene_chain[mutate_pos]
            if original in GENE_COMPLEMENTS:
                self.gene_chain[mutate_pos] = GENE_COMPLEMENTS[original]
    
    def evaluate_decision_quality(self):
        """评估决策质量"""
        if not self.current_piece or self.decisions_made < 2:
            return 0.5
            
        # 计算当前状态
        current_height = self.visual_system._calculate_board_height(self.board)
        current_holes = self.visual_system._count_holes(self.board)
        
        # 计算变化
        if hasattr(self, 'prev_height') and hasattr(self, 'prev_holes'):
            height_change = current_height - self.prev_height
            holes_change = current_holes - self.prev_holes
            
            # 评估质量
            quality = 0.5  # 基准分
            
            # 高度降低是好的
            if height_change < 0:
                quality += 0.2
            elif height_change > 2:  # 高度大幅增加是坏的
                quality -= 0.2
            
            # 空洞减少是好的
            if holes_change < 0:
                quality += 0.15
            elif holes_change > 1:  # 空洞增加是坏的
                quality -= 0.15
            
            # 如果有消除行，非常好
            if self.lines_cleared > self.prev_lines if hasattr(self, 'prev_lines') else 0:
                quality += 0.3
        else:
            quality = 0.5
        
        # 保存当前状态
        self.prev_height = current_height
        self.prev_holes = current_holes
        self.prev_lines = self.lines_cleared
        
        return max(0.0, min(1.0, quality))
    
    def ged_decision(self):
        """GEDA决策过程（完整保留原算法流程）"""
        self.decisions_made += 1
        
        # 更新视觉系统
        local_genes = self.visual_system.update_visual_input(
            self.board, self.current_piece, self.next_piece, self.gene_chain
        )
        
        # 1. 计算环境压力
        pressure = self.calculate_environment_pressure()
        
        # 2. 提取环境特征
        features = self.extract_environment_features()
        features_key = tuple(round(f, 2) for f in features[:6])
        
        # 3. 检查缓存
        if features_key in self.expression_cache:
            mode = self.expression_cache[features_key]
        else:
            # 4. 激活基因段
            gene_segment, segment_type = self.activate_gene_segments(features)
            
            # 5. 基因表达为动作
            mode = self.map_gene_to_action(gene_segment, features)
            
            # 6. 缓存结果
            self.expression_cache[features_key] = mode
        
        # 7. 评估上次决策的质量
        if self.decisions_made > 1:
            quality = self.evaluate_decision_quality()
            self.decision_quality_history.append(quality)
            
            if quality > 0.7:
                self.successful_decisions += 1
                self.consecutive_successes += 1
                self.consecutive_failures = max(0, self.consecutive_failures - 1)
                
                # 成功决策强化基因
                if random.random() < 0.4:
                    self.mutate_gene_chain('insert' if random.random() < 0.5 else 'replace')
            elif quality < 0.3:
                self.consecutive_successes = max(0, self.consecutive_successes - 1)
                self.consecutive_failures += 1
                
                # 失败决策可能触发变异
                if random.random() < 0.3:
                    self.mutate_gene_chain('replace' if random.random() < 0.7 else 'complement')
        
        self.last_decision_time = time.time()
        return mode, pressure
    
    def is_valid_position(self, piece, x=None, y=None, rotation=None):
        """检查位置是否有效"""
        if x is None:
            x = piece.x
        if y is None:
            y = piece.y
        if rotation is None:
            rotation = piece.rotation
        
        # 获取旋转后的形状
        shape = piece.shape
        for _ in range(rotation):
            shape = [[shape[y][x] for y in range(len(shape)-1, -1, -1)] 
                    for x in range(len(shape[0]))]
        
        # 检查每个方块
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col]:
                    board_x = x + col
                    board_y = y + row
                    
                    if (board_x < 0 or board_x >= GRID_WIDTH or
                        board_y < 0 or board_y >= GRID_HEIGHT or
                        self.board[board_y][board_x] != 0):
                        return False
        return True
    
    def lock_piece(self):
        """锁定当前方块到棋盘"""
        if not self.current_piece:
            return
            
        shape = self.current_piece.get_rotated_shape()
        
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col]:
                    board_y = self.current_piece.y + row
                    board_x = self.current_piece.x + col
                    if 0 <= board_y < GRID_HEIGHT and 0 <= board_x < GRID_WIDTH:
                        self.board[board_y][board_x] = self.current_piece.type + 1
        
        # 检查消除行
        self.clear_lines()
        
        # 生成新方块
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()
        self.pieces_placed += 1
        
        # 重置自动移动
        self.move_direction = 0
        self.rotate_needed = False
        self.hard_drop = False
        self.target_position = None
        self.move_complete = False
        
        # 检查游戏是否结束
        if not self.is_valid_position(self.current_piece):
            self.game_over = True
        
        # 记录成功决策
        if self.lines_cleared > self.prev_lines if hasattr(self, 'prev_lines') else 0:
            self.successful_decisions += 1
    
    def clear_lines(self):
        """消除完整的行"""
        lines_to_clear = []
        
        for y in range(GRID_HEIGHT):
            if all(self.board[y][x] != 0 for x in range(GRID_WIDTH)):
                lines_to_clear.append(y)
        
        # 如果没有行要消除，直接返回
        if not lines_to_clear:
            return
        
        # 消除行
        for line in lines_to_clear:
            # 移除该行
            del self.board[line]
            # 在顶部添加新行
            self.board.insert(0, [0 for _ in range(GRID_WIDTH)])
        
        # 更新分数
        lines_count = len(lines_to_clear)
        self.lines_cleared += lines_count
        
        # 计算分数（俄罗斯方块标准计分）
        line_scores = {1: 100, 2: 300, 3: 500, 4: 800}
        self.score += line_scores.get(lines_count, 0) * self.level
        
        # 更新等级（每10行升一级）
        self.level = self.lines_cleared // 10 + 1
        
        # 更新下落速度（随等级增加而加快）
        self.fall_speed = max(0.05, 0.5 - (self.level - 1) * 0.05)
        
        # 记录环境历史
        self.environment_history.append({
            'lines': lines_count,
            'score': self.score,
            'level': self.level,
            'timestamp': time.time()
        })
        
        # 成功消除行，可能触发基因强化
        if lines_count >= 2 and random.random() < 0.6:
            self.mutate_gene_chain('insert' if random.random() < 0.5 else 'replace')
    
    def execute_auto_moves(self):
        """执行自动移动"""
        current_time = time.time()
        
        # 检查是否需要执行移动
        if current_time - self.last_move_time < self.auto_move_delay:
            return
        
        self.last_move_time = current_time
        
        # 旋转
        if self.rotate_needed:
            old_rotation = self.current_piece.rotation
            self.current_piece.rotate()
            if not self.is_valid_position(self.current_piece):
                self.current_piece.rotation = old_rotation
            self.rotate_needed = False
        
        # 水平移动
        if self.move_direction != 0:
            old_x = self.current_piece.x
            self.current_piece.x += self.move_direction
            if not self.is_valid_position(self.current_piece):
                self.current_piece.x = old_x
                self.move_direction = 0
        
        # 硬降
        if self.hard_drop:
            # 一直下落直到碰撞
            while self.is_valid_position(self.current_piece, y=self.current_piece.y + 1):
                self.current_piece.y += 1
            
            # 锁定方块
            self.lock_piece()
            self.hard_drop = False
            self.move_complete = True
    
    def update(self, dt):
        """更新游戏状态 - 完全自动运行"""
        if self.game_over:
            return
        
        # 做出GEDA决策
        if not self.move_complete:
            decision_mode, pressure = self.ged_decision()
        
        # 执行自动移动
        self.execute_auto_moves()
        
        # 方块自然下落
        current_time = time.time()
        if current_time - self.last_fall_time > self.fall_speed:
            # 尝试下落
            self.current_piece.y += 1
            
            # 检查是否碰撞
            if not self.is_valid_position(self.current_piece):
                # 回退并锁定
                self.current_piece.y -= 1
                self.lock_piece()
            
            self.last_fall_time = current_time
        
        # 定期变异
        if random.random() < 0.05:
            self.mutate_gene_chain()
        
        # 更新记忆数据库
        self.update_memory_dbs()
        
        # 调整探索率
        avg_quality = sum(self.decision_quality_history) / max(len(self.decision_quality_history), 1)
        if avg_quality > 0.7:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
        elif avg_quality < 0.3:
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
    
    def update_memory_dbs(self):
        """更新记忆数据库"""
        if self.current_piece:
            key = (self.current_piece.x, self.current_piece.y)
            
            # 记录基因表达信息
            gene_expr = self.visual_system.get_gene_activation_vector(
                (self.current_piece.x, self.current_piece.y)
            )
            
            self.memory_dbs[key] = {
                'score': self.score,
                'lines': self.lines_cleared,
                'level': self.level,
                'gene_length': len(self.gene_chain),
                'gene_expression': gene_expr.tolist(),
                'pressure': self.calculate_environment_pressure(),
                'timestamp': time.time()
            }
    
    def draw(self, screen):
        """绘制游戏"""
        # 绘制背景
        screen.fill(BLACK)
        
        # 绘制游戏区域边框
        border_rect = pygame.Rect(
            0, 0,
            GRID_WIDTH * BLOCK_SIZE + 2,
            GRID_HEIGHT * BLOCK_SIZE + 2
        )
        pygame.draw.rect(screen, WHITE, border_rect, 2)
        
        # 绘制已锁定的方块
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.board[y][x] != 0:
                    color_idx = self.board[y][x] - 1
                    if color_idx < len(COLORS):
                        color = COLORS[color_idx]
                    else:
                        color = GRAY
                    
                    pygame.draw.rect(screen, color, 
                                    (x * BLOCK_SIZE, y * BLOCK_SIZE, 
                                     BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(screen, WHITE, 
                                    (x * BLOCK_SIZE, y * BLOCK_SIZE, 
                                     BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # 绘制当前方块
        if self.current_piece and not self.game_over:
            shape = self.current_piece.get_rotated_shape()
            color = self.current_piece.color
            
            for row in range(len(shape)):
                for col in range(len(shape[0])):
                    if shape[row][col]:
                        pygame.draw.rect(screen, color, 
                                        ((self.current_piece.x + col) * BLOCK_SIZE,
                                         (self.current_piece.y + row) * BLOCK_SIZE,
                                         BLOCK_SIZE, BLOCK_SIZE))
                        pygame.draw.rect(screen, WHITE, 
                                        ((self.current_piece.x + col) * BLOCK_SIZE,
                                         (self.current_piece.y + row) * BLOCK_SIZE,
                                         BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # 绘制网格线
        for x in range(0, GRID_WIDTH * BLOCK_SIZE, BLOCK_SIZE):
            pygame.draw.line(screen, DARK_GRAY, (x, 0), (x, GRID_HEIGHT * BLOCK_SIZE), 1)
        for y in range(0, GRID_HEIGHT * BLOCK_SIZE, BLOCK_SIZE):
            pygame.draw.line(screen, DARK_GRAY, (0, y), (GRID_WIDTH * BLOCK_SIZE, y), 1)
        
        # 绘制视觉模拟区域
        visual_offset_x = GRID_WIDTH * BLOCK_SIZE + 20
        
        # 绘制基因表达地图
        self.visual_system.render_gene_expression_map(
            screen, visual_offset_x, 10, 5
        )
        
        # 绘制环境压力地图
        self.visual_system.render_pressure_map(
            screen, visual_offset_x, 
            GRID_HEIGHT * 5 + 40,
            5
        )
        
        # 绘制信息面板
        info_x = visual_offset_x + VISUAL_WIDTH * 5 + 20
        
        # 游戏状态
        font = pygame.font.Font(None, 28)
        large_font = pygame.font.Font(None, 36)
        
        # 分数和等级
        score_text = large_font.render(f"分数: {self.score}", True, WHITE)
        screen.blit(score_text, (info_x, 10))
        
        level_text = font.render(f"等级: {self.level}", True, YELLOW)
        screen.blit(level_text, (info_x, 50))
        
        lines_text = font.render(f"消除行: {self.lines_cleared}", True, CYAN)
        screen.blit(lines_text, (info_x, 80))
        
        # 下一个方块预览
        next_text = font.render("下一个:", True, WHITE)
        screen.blit(next_text, (info_x, 120))
        
        if self.next_piece:
            preview_x = info_x + 40
            preview_y = 150
            
            # 绘制预览背景
            pygame.draw.rect(screen, DARK_GRAY, 
                           (preview_x - 10, preview_y - 10, 
                            len(self.next_piece.shape[0]) * 20 + 20,
                            len(self.next_piece.shape) * 20 + 20))
            
            # 绘制预览方块
            for row in range(len(self.next_piece.shape)):
                for col in range(len(self.next_piece.shape[0])):
                    if self.next_piece.shape[row][col]:
                        pygame.draw.rect(screen, self.next_piece.color,
                                        (preview_x + col * 20, preview_y + row * 20, 18, 18))
        
        # 基因信息
        gene_title = font.render("GEDA基因系统", True, PURPLE)
        screen.blit(gene_title, (info_x, 220))
        
        gene_text = font.render(f"基因长度: {len(self.gene_chain)}", True, YELLOW)
        screen.blit(gene_text, (info_x, 250))
        
        # 决策信息
        success_rate = self.successful_decisions / max(self.decisions_made, 1)
        success_text = font.render(f"成功率: {success_rate:.2f}", True, 
                                  GREEN if success_rate > 0.7 else RED if success_rate < 0.3 else YELLOW)
        screen.blit(success_text, (info_x, 280))
        
        # 压力信息
        pressure = self.calculate_environment_pressure()
        pressure_text = font.render(f"环境压力: {pressure:.2f}", True, 
                                   RED if pressure > 0.7 else GREEN if pressure < 0.3 else YELLOW)
        screen.blit(pressure_text, (info_x, 310))
        
        # 变异信息
        mutate_text = font.render(f"变异次数: {self.mutations_count}", True, PURPLE)
        screen.blit(mutate_text, (info_x, 340))
        
        # 基因表达信息
        expr_text = font.render(f"基因表达: {self.gene_expressions}", True, CYAN)
        screen.blit(expr_text, (info_x, 370))
        
        # 决策模式
        mode_text = font.render(f"决策模式: {self.decision_mode}", True, 
                               RED if self.decision_mode == "激进模式" else 
                               GREEN if self.decision_mode == "保守模式" else 
                               MAGENTA if self.decision_mode == "探索模式" else 
                               YELLOW)
        screen.blit(mode_text, (info_x, 400))
        
        # 目标位置信息
        if self.target_position:
            target_text = font.render(f"目标: ({self.target_position[0]}, {self.target_position[1]})", 
                                     True, LIGHT_BLUE)
            screen.blit(target_text, (info_x, 430))
        
        # 显示当前基因片段
        if len(self.gene_chain) > 0:
            sample_start = random.randint(0, max(0, len(self.gene_chain) - 15))
            gene_sample = ''.join(self.gene_chain[sample_start:sample_start+15])
            gene_display = font.render(f"基因: {gene_sample}", True, YELLOW)
            screen.blit(gene_display, (info_x, 460))
            
            # 显示基因组成
            gene_counts = {'A': 0, 'G': 0, 'C': 0, 'T': 0, 'X': 0}
            for gene in self.gene_chain[:30]:
                if gene in gene_counts:
                    gene_counts[gene] += 1
            
            total = sum(gene_counts.values())
            if total > 0:
                composition_text = font.render(
                    f"A:{gene_counts['A']/total:.1%} G:{gene_counts['G']/total:.1%} "
                    f"C:{gene_counts['C']/total:.1%} T:{gene_counts['T']/total:.1%} "
                    f"X:{gene_counts['X']/total:.1%}", 
                    True, CYAN
                )
                screen.blit(composition_text, (info_x, 490))
        
        # 操作说明
        controls_y = 530
        controls = [
            "GEDA俄罗斯方块: 完全自动运行",
            "A:激进 G:稳定 C:保守 T:灵活 X:探索",
            "环境压力控制基因表达类型",
            "基因记忆系统记录成功模式",
            "R键: 重新开始（新一代进化）",
            "空格键: 暂停/继续（仅观看）"
        ]
        
        for i, control in enumerate(controls):
            control_text = font.render(control, True, LIGHT_GRAY)
            screen.blit(control_text, (info_x, controls_y + i * 25))
        
        # 如果游戏结束，显示游戏结束文字
        if self.game_over:
            game_over_font = pygame.font.Font(None, 48)
            game_over_text = game_over_font.render("GEDA 游戏结束", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            screen.blit(game_over_text, text_rect)
            
            restart_text = large_font.render("按R键开始新一代进化", True, YELLOW)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
            screen.blit(restart_text, restart_rect)
            
            # 显示最终统计
            stats_text = font.render(
                f"最终基因长度: {len(self.gene_chain)} | "
                f"变异次数: {self.mutations_count} | "
                f"表达次数: {self.gene_expressions}", 
                True, WHITE
            )
            stats_rect = stats_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 90))
            screen.blit(stats_text, stats_rect)

class GEDATetrisGame:
    """GEDA俄罗斯方块游戏 - 完全自动运行"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("GEDA Tetris - 基因表达决策算法（完全自动版）")
        self.clock = pygame.time.Clock()
        self.game = GEDATetris()
        self.running = True
        self.paused = False
        self.generation = 1
        self.total_score = 0
        self.highest_score = 0
        
        # 统计信息
        self.stats_history = deque(maxlen=20)
        
        print("=" * 70)
        print("GEDA Tetris - 基因表达决策算法（完全自动版）")
        print("=" * 70)
        print("核心特性:")
        print("1. 五进制碱基系统: A(激进行动)/G(稳定操作)/C(保守策略)/T(灵活调整)/X(突变探索)")
        print("2. 完整视觉系统: 实时基因表达地图和环境压力地图")
        print("3. 环境压力感知: 动态计算压力值，控制基因表达类型")
        print("4. 智能位置评估: 多维度评估最佳放置位置")
        print("5. 动态进化: 5种变异类型，基因链自然生长")
        print("6. 完全自动运行: 算法自主决策，无需人工干预")
        print("=" * 70)
    
    def run(self):
        """运行游戏主循环 - 完全自动运行"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 处理事件（仅暂停和重新开始）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print(f"游戏{'暂停' if self.paused else '继续'}")
                    elif event.key == pygame.K_r:
                        # 记录当前代的统计信息
                        self.record_generation_stats()
                        
                        # 创建新的一代（继承基因）
                        inherited_genes = self.game.gene_chain.copy() if self.game.score > 100 else None
                        self.game = GEDATetris(inherited_genes)
                        self.generation += 1
                        print(f"\n第 {self.generation} 代开始...")
            
            if not self.paused and not self.game.game_over:
                # 更新游戏状态（完全自动）
                self.game.update(dt)
                
                # 更新最高分
                if self.game.score > self.highest_score:
                    self.highest_score = self.game.score
            
            # 绘制游戏
            self.screen.fill(BLACK)
            self.game.draw(self.screen)
            
            # 显示代数信息
            font = pygame.font.Font(None, 36)
            gen_text = font.render(f"Generation: {self.generation}", True, PURPLE)
            self.screen.blit(gen_text, (SCREEN_WIDTH - 200, 10))
            
            # 显示最高分
            high_score_text = font.render(f"最高分: {self.highest_score}", True, YELLOW)
            self.screen.blit(high_score_text, (SCREEN_WIDTH - 200, 50))
            
            # 显示暂停信息
            if self.paused:
                pause_font = pygame.font.Font(None, 48)
                pause_text = pause_font.render("游戏暂停", True, YELLOW)
                text_rect = pause_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
                self.screen.blit(pause_text, text_rect)
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        self.print_final_stats()
    
    def record_generation_stats(self):
        """记录一代的统计信息"""
        stats = {
            'generation': self.generation,
            'score': self.game.score,
            'lines_cleared': self.game.lines_cleared,
            'level': self.game.level,
            'gene_length': len(self.game.gene_chain),
            'mutations': self.game.mutations_count,
            'expressions': self.game.gene_expressions,
            'decisions': self.game.decisions_made,
            'success_rate': self.game.successful_decisions / max(self.game.decisions_made, 1)
        }
        
        self.stats_history.append(stats)
        self.total_score += self.game.score
        
        # 打印本代统计
        print(f"第 {self.generation} 代结束:")
        print(f"  分数: {self.game.score}")
        print(f"  消除行数: {self.game.lines_cleared}")
        print(f"  等级: {self.game.level}")
        print(f"  基因长度: {len(self.game.gene_chain)}")
        print(f"  变异次数: {self.game.mutations_count}")
        print(f"  决策成功率: {stats['success_rate']:.2%}")
        
        # 显示基因组成
        gene_counts = {'A': 0, 'G': 0, 'C': 0, 'T': 0, 'X': 0}
        for gene in self.game.gene_chain:
            if gene in gene_counts:
                gene_counts[gene] += 1
        
        total = sum(gene_counts.values())
        if total > 0:
            print(f"  基因组成: A:{gene_counts['A']/total:.1%} G:{gene_counts['G']/total:.1%} "
                  f"C:{gene_counts['C']/total:.1%} T:{gene_counts['T']/total:.1%} "
                  f"X:{gene_counts['X']/total:.1%}")
    
    def print_final_stats(self):
        """打印最终统计信息"""
        print("\n" + "=" * 70)
        print("GEDA Tetris - 最终统计")
        print("=" * 70)
        print(f"总代数: {self.generation}")
        print(f"最高分数: {self.highest_score}")
        print(f"总分数: {self.total_score}")
        print(f"平均分数: {self.total_score/max(self.generation, 1):.1f}")
        
        if self.stats_history:
            # 计算平均统计
            avg_lines = sum(s['lines_cleared'] for s in self.stats_history) / len(self.stats_history)
            avg_level = sum(s['level'] for s in self.stats_history) / len(self.stats_history)
            avg_gene_length = sum(s['gene_length'] for s in self.stats_history) / len(self.stats_history)
            avg_mutations = sum(s['mutations'] for s in self.stats_history) / len(self.stats_history)
            avg_success_rate = sum(s['success_rate'] for s in self.stats_history) / len(self.stats_history) * 100
            
            print(f"平均消除行数: {avg_lines:.1f}")
            print(f"平均等级: {avg_level:.1f}")
            print(f"平均基因长度: {avg_gene_length:.1f}")
            print(f"平均变异次数: {avg_mutations:.1f}")
            print(f"平均成功率: {avg_success_rate:.1f}%")
        
        print("\n算法总结:")
        print("1. 完整GEDA算法: 五进制碱基系统，基因表达，环境压力感知")
        print("2. 视觉系统: 实时显示基因表达和环境压力")
        print("3. 完全自动运行: 算法自主决策，无需人工干预")
        print("4. 强化学习: 根据决策质量调整探索率和变异策略")
        print("5. 多代进化: 优秀基因得以继承和优化")
        print("=" * 70)

# 运行游戏
if __name__ == "__main__":
    game = GEDATetrisGame()
    game.run()