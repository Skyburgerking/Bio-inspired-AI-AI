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

# 游戏参数
GRID_SIZE = 20
GRID_WIDTH = 25
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH + 450  # 增加视觉模拟区域
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT + 180
FPS = 10

# 方向常量
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# 视觉模拟常量
VISION_RADIUS = 6
COGNITIVE_MAP_SCALE = 5

# GEDA碱基常量
GENE_BASES = ['A', 'G', 'C', 'T', 'X']  # 激进行动/稳定操作/保守策略/灵活调整/突变探索
GENE_COMPLEMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'X': 'X'}

class GEDAVisualSystem:
    """GEDA视觉模拟系统"""
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # 视觉层：当前视野
        self.visual_layer = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # 基因表达层：基因激活状态
        self.gene_expression_layer = np.zeros((grid_height, grid_width, 4), dtype=np.float32)
        # 通道0: A碱基(激进)激活度
        # 通道1: G碱基(稳定)激活度
        # 通道2: C碱基(保守)激活度
        # 通道3: T碱基(灵活)激活度
        
        # X碱基(探索)层
        self.exploration_layer = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # 环境压力层
        self.pressure_layer = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # 基因记忆库：位置->基因表达记录
        self.gene_memory = {}
        
        # 视野记忆
        self.vision_memory = deque(maxlen=20)
        
        # 路径规划器
        self.pathfinder = AStarPathfinder(grid_width, grid_height)
        
        # 初始化基因表达模式
        self._initialize_gene_patterns()
    
    def _initialize_gene_patterns(self):
        """初始化基因表达模式"""
        # 边界区域：高C碱基表达（保守）
        self.gene_expression_layer[0, :, 2] = 0.8  # 上边界
        self.gene_expression_layer[-1, :, 2] = 0.8  # 下边界
        self.gene_expression_layer[:, 0, 2] = 0.8  # 左边界
        self.gene_expression_layer[:, -1, 2] = 0.8  # 右边界
        
        # 中心区域：中等G碱基表达（稳定）
        center_y = self.grid_height // 2
        center_x = self.grid_width // 2
        for y in range(max(0, center_y-2), min(self.grid_height, center_y+3)):
            for x in range(max(0, center_x-2), min(self.grid_width, center_x+3)):
                self.gene_expression_layer[y, x, 1] = 0.5
    
    def update_visual_input(self, snake_head, snake_body, food_pos, gene_chain):
        """更新视觉输入和基因表达"""
        # 清空视觉层
        self.visual_layer.fill(0)
        
        # 清空探索层
        self.exploration_layer.fill(0)
        
        # 计算当前位置的基因表达特征
        head_x, head_y = snake_head
        local_genes = self._extract_local_genes(gene_chain, head_x, head_y)
        
        # 更新基因表达层
        self._update_gene_expression(snake_head, snake_body, food_pos, local_genes)
        
        # 更新视野
        self._update_visual_field(snake_head, snake_body, food_pos)
        
        # 更新环境压力层
        self._update_pressure_layer(snake_head, snake_body, food_pos)
        
        # 更新X碱基探索层
        self._update_exploration_layer(snake_head, local_genes)
        
        # 记录视觉记忆
        self.vision_memory.append({
            'position': snake_head,
            'genes': local_genes,
            'visual': self.visual_layer.copy(),
            'pressure': self.pressure_layer[head_y, head_x]
        })
    
    def _extract_local_genes(self, gene_chain, x, y):
        """根据位置提取局部基因特征"""
        # 使用位置信息作为种子来提取基因
        seed = (x * 17 + y * 31) % max(len(gene_chain), 1)
        gene_window = 5
        
        local_genes = []
        for i in range(gene_window):
            idx = (seed + i) % len(gene_chain)
            local_genes.append(gene_chain[idx])
        
        return local_genes
    
    def _update_gene_expression(self, snake_head, snake_body, food_pos, local_genes):
        """更新基因表达层"""
        head_x, head_y = snake_head
        food_x, food_y = food_pos
        
        # 计算环境压力值
        pressure = self._calculate_local_pressure(snake_head, snake_body, food_pos)
        
        # 根据压力值调整基因表达
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # 距离蛇头的距离因子
                dist_to_head = math.sqrt((x - head_x)**2 + (y - head_y)**2)
                distance_factor = max(0, 1 - dist_to_head / VISION_RADIUS)
                
                # 距离食物的因子
                dist_to_food = abs(x - food_x) + abs(y - food_y)
                food_factor = max(0, 1 - dist_to_food / (self.grid_width + self.grid_height))
                
                # 基因表达计算
                if distance_factor > 0:  # 在视野范围内
                    # 分析局部基因组成
                    gene_counts = {'A': 0, 'G': 0, 'C': 0, 'T': 0, 'X': 0}
                    for gene in local_genes:
                        gene_counts[gene] += 1
                    
                    total = len(local_genes)
                    
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
                        self.exploration_layer[y, x, 0] = int(255 * explore_intensity)  # 红色通道表示探索
                
                # 记忆位置基因表达
                if (x, y) == snake_head:
                    self.gene_memory[(x, y)] = {
                        'genes': local_genes,
                        'pressure': pressure,
                        'expression': self.gene_expression_layer[y, x].copy(),
                        'timestamp': time.time()
                    }
    
    def _calculate_local_pressure(self, snake_head, snake_body, food_pos):
        """计算局部环境压力"""
        head_x, head_y = snake_head
        food_x, food_y = food_pos
        
        # 1. 饥饿压力（简化）
        hunger_pressure = 0.3
        
        # 2. 空间压力
        obstacles = 0
        for dx, dy in [UP, DOWN, LEFT, RIGHT]:
            nx, ny = head_x + dx, head_y + dy
            if (nx < 0 or nx >= self.grid_width or 
                ny < 0 or ny >= self.grid_height or 
                (nx, ny) in snake_body):
                obstacles += 1
        
        space_pressure = obstacles / 4
        
        # 3. 食物接近压力
        dist_to_food = abs(head_x - food_x) + abs(head_y - food_y)
        food_pressure = 1.0 - min(dist_to_food / (self.grid_width + self.grid_height), 1.0)
        
        # 综合压力
        total_pressure = 0.3 * hunger_pressure + 0.4 * space_pressure + 0.3 * food_pressure
        return total_pressure
    
    def _update_visual_field(self, snake_head, snake_body, food_pos):
        """更新视野层"""
        head_x, head_y = snake_head
        
        # 计算视野范围
        vision_start_x = max(0, head_x - VISION_RADIUS)
        vision_end_x = min(self.grid_width, head_x + VISION_RADIUS + 1)
        vision_start_y = max(0, head_y - VISION_RADIUS)
        vision_end_y = min(self.grid_height, head_y + VISION_RADIUS + 1)
        
        # 在视野范围内绘制
        for y in range(vision_start_y, vision_end_y):
            for x in range(vision_start_x, vision_end_x):
                # 计算到蛇头的距离
                dist = math.sqrt((x - head_x)**2 + (y - head_y)**2)
                if dist <= VISION_RADIUS:
                    # 食物
                    if (x, y) == food_pos:
                        self.visual_layer[y, x] = [255, 50, 50]  # 亮红色
                    
                    # 蛇身
                    elif (x, y) in snake_body:
                        idx = snake_body.index((x, y))
                        if idx == 0:  # 蛇头
                            self.visual_layer[y, x] = [0, 255, 0]  # 绿色
                        else:  # 蛇身
                            body_ratio = idx / len(snake_body)
                            green_value = int(255 * (1 - body_ratio * 0.5))
                            self.visual_layer[y, x] = [0, green_value, 100]
                    
                    # 空位置
                    else:
                        # 根据基因表达着色
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
                            gray_value = 80
                            self.visual_layer[y, x] = [gray_value, gray_value, gray_value]
    
    def _update_pressure_layer(self, snake_head, snake_body, food_pos):
        """更新环境压力层"""
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # 计算每个位置的压力
                if (x, y) in snake_body:
                    self.pressure_layer[y, x] = 0.9  # 蛇身位置高压
                elif x == 0 or x == self.grid_width-1 or y == 0 or y == self.grid_height-1:
                    self.pressure_layer[y, x] = 0.8  # 边界高压
                else:
                    # 距离蛇头和食物的压力
                    dist_to_head = math.sqrt((x - snake_head[0])**2 + (y - snake_head[1])**2)
                    dist_to_food = abs(x - food_pos[0]) + abs(y - food_pos[1])
                    
                    head_pressure = max(0, 1 - dist_to_head / (self.grid_width * 0.7))
                    food_pressure = max(0, 1 - dist_to_food / (self.grid_width + self.grid_height))
                    
                    self.pressure_layer[y, x] = (head_pressure + food_pressure) / 2
    
    def _update_exploration_layer(self, snake_head, local_genes):
        """更新探索层（X碱基表达）"""
        head_x, head_y = snake_head
        
        # 检查是否有X碱基
        if 'X' in local_genes:
            x_count = local_genes.count('X')
            explore_intensity = x_count / len(local_genes)
            
            # 在探索层中标记探索区域
            explore_radius = int(VISION_RADIUS * explore_intensity)
            
            for dy in range(-explore_radius, explore_radius+1):
                for dx in range(-explore_radius, explore_radius+1):
                    x, y = head_x + dx, head_y + dy
                    if (0 <= x < self.grid_width and 0 <= y < self.grid_height and
                        dx*dx + dy*dy <= explore_radius*explore_radius):
                        
                        # 根据距离设置探索强度
                        dist = math.sqrt(dx*dx + dy*dy)
                        intensity = max(0, 1 - dist / explore_radius) if explore_radius > 0 else 0
                        
                        # 探索层使用紫色表示探索区域
                        self.exploration_layer[y, x, 0] = int(255 * intensity * 0.7)  # R
                        self.exploration_layer[y, x, 2] = int(255 * intensity)  # B
    
    def get_gene_activation_vector(self, position):
        """获取位置上的基因激活向量"""
        x, y = position
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.gene_expression_layer[y, x].copy()
        return np.zeros(4)
    
    def get_path_direction(self, snake_head, snake_body, food_pos):
        """获取A*路径推荐方向"""
        obstacles = snake_body[1:]  # 排除头部
        return self.pathfinder.get_recommended_direction(snake_head, food_pos, obstacles)
    
    def render_gene_expression_map(self, surface, x_offset, y_offset, scale):
        """渲染基因表达地图"""
        # 创建临时表面
        map_surface = pygame.Surface((self.grid_width * scale, self.grid_height * scale))
        
        # 绘制基因表达地图
        for y in range(self.grid_height):
            for x in range(self.grid_width):
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
        """渲染环境压力地图"""
        # 创建临时表面
        map_surface = pygame.Surface((self.grid_width * scale, self.grid_height * scale))
        
        # 绘制压力地图
        for y in range(self.grid_height):
            for x in range(self.grid_width):
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

class AStarPathfinder:
    """A*寻路算法"""
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def find_path(self, start, goal, obstacles):
        """使用A*算法寻找路径"""
        if start == goal:
            return [start]
        
        obstacle_set = set(obstacles)
        if goal in obstacle_set:
            return []
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < self.grid_width and 
                    0 <= neighbor[1] < self.grid_height and 
                    neighbor not in obstacle_set):
                    
                    tentative_g_score = g_score[current] + 1
                    
                    if (neighbor not in g_score or 
                        tentative_g_score < g_score[neighbor]):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        
                        if neighbor not in [item[1] for item in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def get_recommended_direction(self, start, goal, obstacles):
        """获取推荐移动方向"""
        path = self.find_path(start, goal, obstacles)
        
        if len(path) >= 1:
            next_pos = path[0]
            dx = next_pos[0] - start[0]
            dy = next_pos[1] - start[1]
            
            if dx == 1:
                return RIGHT
            elif dx == -1:
                return LEFT
            elif dy == 1:
                return DOWN
            elif dy == -1:
                return UP
        
        return None

class GEDASnake:
    """GEDA视觉增强版贪吃蛇"""
    def __init__(self):
        # 蛇的初始状态
        self.length = 3
        self.positions = [(GRID_WIDTH//2, GRID_HEIGHT//2)]
        for i in range(1, self.length):
            self.positions.append((GRID_WIDTH//2 - i, GRID_HEIGHT//2))
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.score = 0
        self.steps_without_food = 0
        self.alive = True
        self.food_eaten = 0
        
        # GEDA 基因系统（保持原算法）
        self.gene_chain = self.initialize_gene_chain()
        self.memory_dbs = {}  # 分段记忆数据库
        self.expression_cache = {}
        self.environment_history = deque(maxlen=100)
        
        # 视觉系统（新增）
        self.visual_system = GEDAVisualSystem(GRID_WIDTH, GRID_HEIGHT)
        
        # 性能统计
        self.decisions_made = 0
        self.successful_decisions = 0
        self.gene_expressions = 0
        self.mutations_count = 0
        
        # 食物
        self.food_pos = self.generate_food()
    
    def initialize_gene_chain(self):
        """初始化基础基因链"""
        # 创建更复杂的基因链
        gene_length = 50
        gene_chain = []
        
        # 不同基因段有不同特性
        segments = [
            (10, ['A', 'A', 'G', 'T', 'C', 'A', 'G', 'X', 'T', 'A']),  # 激进段
            (15, ['G', 'C', 'T', 'G', 'C', 'G', 'T', 'G', 'C', 'T', 'G', 'C', 'T', 'G', 'C']),  # 稳定段
            (10, ['C', 'C', 'T', 'A', 'C', 'C', 'T', 'A', 'C', 'C']),  # 保守段
            (8, ['T', 'A', 'G', 'C', 'T', 'A', 'G', 'C']),  # 灵活段
            (7, ['X', 'A', 'X', 'G', 'X', 'C', 'X'])  # 探索段
        ]
        
        for segment_length, segment_pattern in segments:
            for i in range(segment_length):
                gene_chain.append(segment_pattern[i % len(segment_pattern)])
        
        return gene_chain[:gene_length]
    
    def calculate_environment_pressure(self):
        """计算环境压力值（视觉增强版）"""
        head = self.positions[0]
        
        # 使用视觉系统的压力层
        visual_pressure = self.visual_system.pressure_layer[head[1], head[0]]
        
        # 原始压力计算
        hunger_pressure = min(self.steps_without_food / 50, 1.0)
        
        # 空间压力
        obstacles = 0
        for dx, dy in [UP, DOWN, LEFT, RIGHT]:
            nx, ny = head[0] + dx, head[1] + dy
            if (nx < 0 or nx >= GRID_WIDTH or 
                ny < 0 or ny >= GRID_HEIGHT or 
                (nx, ny) in self.positions):
                obstacles += 1
        space_pressure = obstacles / 4
        
        # 食物压力
        dist_to_food = abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])
        food_pressure = 1.0 - min(dist_to_food / (GRID_WIDTH + GRID_HEIGHT), 1.0)
        
        # 基因多样性压力
        gene_diversity = len(set(self.gene_chain[:20])) / 5  # 归一化到0-1
        diversity_pressure = 1.0 - gene_diversity  # 多样性越低，压力越高
        
        # 综合压力（加入视觉压力）
        total_pressure = (0.2 * hunger_pressure + 
                         0.25 * space_pressure + 
                         0.25 * food_pressure + 
                         0.15 * diversity_pressure +
                         0.15 * visual_pressure)
        
        return min(max(total_pressure, 0.0), 1.0)
    
    def extract_environment_features(self):
        """提取环境特征向量（视觉增强版）"""
        head = self.positions[0]
        features = []
        
        # 视觉特征：8个方向的距离
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), 
                      (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            distance = 0
            x, y = head
            while (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and 
                   (x, y) not in self.positions or (x, y) == head):
                x += dx
                y += dy
                distance += 1
                if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                    break
            features.append(min(distance, 10) / 10)
        
        # 基因表达特征
        gene_expr = self.visual_system.get_gene_activation_vector(head)
        features.extend(gene_expr.tolist())
        
        # 食物方向特征
        food_dx = self.food_pos[0] - head[0]
        food_dy = self.food_pos[1] - head[1]
        features.append(food_dx / GRID_WIDTH)
        features.append(food_dy / GRID_HEIGHT)
        
        # 身体特征
        features.append(min(len(self.positions) / 50, 1.0))
        
        # 成功率特征
        success_rate = self.successful_decisions / max(self.decisions_made, 1)
        features.append(success_rate)
        
        # 环境压力特征
        features.append(self.calculate_environment_pressure())
        
        return features
    
    def activate_gene_segments(self, features):
        """激活匹配的基因段"""
        pressure = self.calculate_environment_pressure()
        
        # 根据压力选择基因段
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
            start_idx = max(0, mid_idx - 5)
            end_idx = min(len(self.gene_chain), mid_idx + 5)
            segment_type = "balanced"
        
        # 提取基因段
        gene_segment = self.gene_chain[start_idx:end_idx]
        
        # 记录激活
        self.gene_expressions += 1
        
        return gene_segment, segment_type
    
    def map_gene_to_action(self, gene_segment, features):
        """将基因序列映射为动作（视觉增强版）"""
        head = self.positions[0]
        
        # 分析基因组成
        gene_counts = {'A': 0, 'G': 0, 'C': 0, 'T': 0, 'X': 0}
        for gene in gene_segment[:15]:  # 考虑前15个碱基
            if gene in gene_counts:
                gene_counts[gene] += 1
        
        total = sum(gene_counts.values())
        if total == 0:
            total = 1
        
        # 基因组成比例
        aggressive_ratio = (gene_counts['A'] + gene_counts['G']) / total
        conservative_ratio = (gene_counts['C'] + gene_counts['T']) / total
        explore_ratio = gene_counts['X'] / total
        
        # 获取A*路径推荐（视觉系统）
        path_direction = self.visual_system.get_path_direction(
            head, self.positions, self.food_pos
        )
        
        # 激进基因主导时：积极寻找食物
        if aggressive_ratio > conservative_ratio and aggressive_ratio > 0.4:
            if path_direction:
                return path_direction
            else:
                # 计算食物方向
                target = self.food_pos
                # 简单朝向食物
                food_dx = target[0] - head[0]
                food_dy = target[1] - head[1]
                
                if abs(food_dx) > abs(food_dy):
                    return RIGHT if food_dx > 0 else LEFT
                else:
                    return DOWN if food_dy > 0 else UP
        
        # 探索基因活跃时：随机探索
        elif explore_ratio > 0.2:
            # X碱基带来随机探索
            possible_directions = []
            for dx, dy in [UP, DOWN, LEFT, RIGHT]:
                nx, ny = head[0] + dx, head[1] + dy
                if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and 
                    (nx, ny) not in self.positions):
                    possible_directions.append((dx, dy))
            
            if possible_directions:
                return random.choice(possible_directions)
        
        # 保守基因主导时：安全第一
        # 检查各个方向的安全性
        safe_directions = []
        
        for dx, dy in [UP, DOWN, LEFT, RIGHT]:
            nx, ny = head[0] + dx, head[1] + dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and 
                (nx, ny) not in self.positions):
                safe_directions.append((dx, dy))
        
        if safe_directions:
            # 如果有多个安全方向，优先考虑路径方向
            if path_direction and path_direction in safe_directions:
                return path_direction
            
            # 否则选择靠近食物的方向
            best_dir = None
            min_dist = float('inf')
            for dx, dy in safe_directions:
                nx, ny = head[0] + dx, head[1] + dy
                dist = abs(nx - self.food_pos[0]) + abs(ny - self.food_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_dir = (dx, dy)
            
            if best_dir:
                return best_dir
            
            return random.choice(safe_directions)
        
        # 如果没有安全方向，保持原方向
        return self.direction
    
    def mutate_gene_chain(self, mutation_type=None):
        """基因变异（增强版）"""
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
    
    def ged_decision(self):
        """GEDA决策过程（视觉增强版）"""
        self.decisions_made += 1
        
        # 更新视觉系统
        self.visual_system.update_visual_input(
            self.positions[0], self.positions, self.food_pos, self.gene_chain
        )
        
        # 1. 计算环境压力
        pressure = self.calculate_environment_pressure()
        
        # 2. 提取环境特征
        features = self.extract_environment_features()
        features_key = tuple(round(f, 2) for f in features[:6])
        
        # 3. 检查缓存
        if features_key in self.expression_cache:
            return self.expression_cache[features_key]
        
        # 4. 激活基因段
        gene_segment, segment_type = self.activate_gene_segments(features)
        
        # 5. 基因表达为动作
        action = self.map_gene_to_action(gene_segment, features)
        
        # 6. 缓存结果
        self.expression_cache[features_key] = (action, segment_type)
        
        return action, segment_type
    
    def generate_food(self):
        """生成食物位置"""
        while True:
            pos = (random.randint(0, GRID_WIDTH-1), 
                   random.randint(0, GRID_HEIGHT-1))
            if pos not in self.positions:
                return pos
    
    def update(self):
        """更新游戏状态"""
        if not self.alive:
            return
        
        # 做出GEDA决策
        action, segment_type = self.ged_decision()
        
        # 防止直接反向移动
        if (action[0] * -1, action[1] * -1) != self.direction:
            self.next_direction = action
        
        # 移动蛇头
        head = self.positions[0]
        new_head = (head[0] + self.next_direction[0], 
                    head[1] + self.next_direction[1])
        
        # 检查碰撞
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or
            new_head in self.positions):
            self.alive = False
            return
        
        # 移动蛇
        self.direction = self.next_direction
        self.positions.insert(0, new_head)
        
        # 检查是否吃到食物
        if new_head == self.food_pos:
            self.score += 10
            self.food_eaten += 1
            self.steps_without_food = 0
            self.food_pos = self.generate_food()
            self.successful_decisions += 1
            
            # 成功决策强化基因
            if random.random() < 0.4:  # 40%概率增强当前基因
                self.mutate_gene_chain('insert' if random.random() < 0.5 else 'replace')
        else:
            # 没有吃到食物，移除尾部
            self.positions.pop()
            self.steps_without_food += 1
            
            # 饥饿惩罚
            if self.steps_without_food > 100:
                if random.random() < 0.3:  # 饥饿时更可能变异
                    self.mutate_gene_chain('replace' if random.random() < 0.7 else 'complement')
        
        # 定期变异
        if random.random() < 0.08:  # 8%的变异概率
            self.mutate_gene_chain()
        
        # 更新记忆数据库
        self.update_memory_dbs()
    
    def update_memory_dbs(self):
        """更新记忆数据库"""
        key = tuple(self.positions[0])  # 使用蛇头位置作为键
        
        # 记录基因表达信息
        gene_expr = self.visual_system.get_gene_activation_vector(self.positions[0])
        
        self.memory_dbs[key] = {
            'score': self.score,
            'food_eaten': self.food_eaten,
            'steps': self.steps_without_food,
            'gene_length': len(self.gene_chain),
            'gene_expression': gene_expr.tolist(),
            'pressure': self.calculate_environment_pressure(),
            'timestamp': time.time()
        }
    
    def draw(self, screen):
        """绘制游戏"""
        # 绘制网格背景
        for x in range(0, GRID_SIZE * GRID_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (x, 0), (x, GRID_HEIGHT * GRID_SIZE), 1)
        for y in range(0, GRID_HEIGHT * GRID_SIZE, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (0, y), (GRID_WIDTH * GRID_SIZE, y), 1)
        
        # 绘制蛇
        for i, (x, y) in enumerate(self.positions):
            # 蛇头为绿色，蛇身为渐变的蓝色
            if i == 0:
                color = GREEN
            else:
                body_ratio = i / len(self.positions)
                blue_value = int(255 * (1 - body_ratio * 0.7))
                color = (0, blue_value, 150)
            
            pygame.draw.rect(screen, color, 
                            (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, WHITE, 
                            (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
        
        # 绘制食物
        pygame.draw.rect(screen, RED, 
                        (self.food_pos[0] * GRID_SIZE, 
                         self.food_pos[1] * GRID_SIZE, 
                         GRID_SIZE, GRID_SIZE))
        
        # 绘制视觉模拟区域
        visual_offset_x = GRID_SIZE * GRID_WIDTH + 10
        
        # 绘制基因表达地图
        self.visual_system.render_gene_expression_map(
            screen, visual_offset_x, 10, COGNITIVE_MAP_SCALE
        )
        
        # 绘制环境压力地图
        self.visual_system.render_pressure_map(
            screen, visual_offset_x, 
            GRID_HEIGHT * COGNITIVE_MAP_SCALE + 40,
            COGNITIVE_MAP_SCALE
        )
        
        # 绘制信息面板
        info_y = GRID_HEIGHT * GRID_SIZE + 10
        font = pygame.font.Font(None, 24)
        large_font = pygame.font.Font(None, 32)
        
        # 游戏状态
        score_text = large_font.render(f"GEDA Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, info_y))
        
        # 基因信息
        gene_text = font.render(f"基因长度: {len(self.gene_chain)}", True, YELLOW)
        screen.blit(gene_text, (10, info_y + 30))
        
        food_text = font.render(f"食物: {self.food_eaten}", True, WHITE)
        screen.blit(food_text, (10, info_y + 55))
        
        # 决策信息
        success_rate = self.successful_decisions / max(self.decisions_made, 1)
        success_text = font.render(f"成功率: {success_rate:.2f}", True, 
                                  GREEN if success_rate > 0.7 else RED if success_rate < 0.3 else YELLOW)
        screen.blit(success_text, (10, info_y + 80))
        
        # 压力信息
        pressure = self.calculate_environment_pressure()
        pressure_text = font.render(f"环境压力: {pressure:.2f}", True, 
                                   RED if pressure > 0.7 else GREEN if pressure < 0.3 else YELLOW)
        screen.blit(pressure_text, (10, info_y + 105))
        
        # 变异信息
        mutate_text = font.render(f"变异次数: {self.mutations_count}", True, PURPLE)
        screen.blit(mutate_text, (10, info_y + 130))
        
        # 基因表达信息
        expr_text = font.render(f"基因表达: {self.gene_expressions}", True, CYAN)
        screen.blit(expr_text, (10, info_y + 155))
        
        # 显示当前基因片段
        if len(self.gene_chain) > 0:
            sample_start = random.randint(0, max(0, len(self.gene_chain) - 15))
            gene_sample = ''.join(self.gene_chain[sample_start:sample_start+15])
            gene_display = font.render(f"基因: {gene_sample}", True, YELLOW)
            screen.blit(gene_display, (200, info_y))
            
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
                screen.blit(composition_text, (200, info_y + 25))
        
        # 操作说明
        controls_y = info_y + 180
        controls = [
            "GEDA算法: 基因表达决策系统",
            "A:激进 G:稳定 C:保守 T:灵活 X:探索",
            "环境压力控制基因表达类型",
            "R键: 重新开始（新一代进化）",
            "空格键: 暂停/继续"
        ]
        
        for i, control in enumerate(controls):
            control_text = font.render(control, True, GRAY)
            screen.blit(control_text, (10, controls_y + i * 20))
        
        # 如果游戏结束，显示游戏结束文字
        if not self.alive:
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

class GEDAGame:
    """GEDA贪吃蛇游戏"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("GEDA Visual Snake - 基因表达决策算法（视觉增强版）")
        self.clock = pygame.time.Clock()
        self.snake = GEDASnake()
        self.running = True
        self.paused = False
        self.generation = 1
        self.total_score = 0
        self.highest_score = 0
        
        # 统计信息
        self.stats_history = deque(maxlen=20)
        
        print("=" * 70)
        print("GEDA Visual Snake - 基因表达决策算法（视觉增强版）")
        print("=" * 70)
        print("核心特性:")
        print("1. 五进制碱基系统: A(激进行动)/G(稳定操作)/C(保守策略)/T(灵活调整)/X(突变探索)")
        print("2. 视觉感知系统: 实时基因表达地图和环境压力地图")
        print("3. 环境压力感知: 动态计算压力值，控制基因表达类型")
        print("4. A*路径规划: 智能寻路与基因决策结合")
        print("5. 动态进化: 5种变异类型，基因链自然生长")
        print("=" * 70)
    
    def run(self):
        """运行游戏主循环"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        # 记录当前代的统计信息
                        self.record_generation_stats()
                        
                        # 创建新的一代
                        self.snake = GEDASnake()
                        self.generation += 1
                        print(f"\n第 {self.generation} 代开始...")
            
            if not self.paused and self.snake.alive:
                # 更新游戏状态
                self.snake.update()
                
                # 更新最高分
                if self.snake.score > self.highest_score:
                    self.highest_score = self.snake.score
            
            # 绘制游戏
            self.screen.fill(BLACK)
            self.snake.draw(self.screen)
            
            # 显示代数信息
            font = pygame.font.Font(None, 36)
            gen_text = font.render(f"Generation: {self.generation}", True, PURPLE)
            self.screen.blit(gen_text, (SCREEN_WIDTH - 200, 10))
            
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
            'score': self.snake.score,
            'food_eaten': self.snake.food_eaten,
            'gene_length': len(self.snake.gene_chain),
            'mutations': self.snake.mutations_count,
            'expressions': self.snake.gene_expressions,
            'decisions': self.snake.decisions_made,
            'success_rate': self.snake.successful_decisions / max(self.snake.decisions_made, 1)
        }
        
        self.stats_history.append(stats)
        self.total_score += self.snake.score
        
        # 打印本代统计
        print(f"第 {self.generation} 代结束:")
        print(f"  分数: {self.snake.score}")
        print(f"  食物: {self.snake.food_eaten}")
        print(f"  基因长度: {len(self.snake.gene_chain)}")
        print(f"  变异次数: {self.snake.mutations_count}")
        print(f"  决策成功率: {stats['success_rate']:.2%}")
        
        # 显示基因组成
        gene_counts = {'A': 0, 'G': 0, 'C': 0, 'T': 0, 'X': 0}
        for gene in self.snake.gene_chain:
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
        print("GEDA Visual Snake - 最终统计")
        print("=" * 70)
        print(f"总代数: {self.generation}")
        print(f"最高分数: {self.highest_score}")
        print(f"总分数: {self.total_score}")
        print(f"平均分数: {self.total_score/max(self.generation, 1):.1f}")
        
        if self.stats_history:
            # 计算平均统计
            avg_gene_length = sum(s['gene_length'] for s in self.stats_history) / len(self.stats_history)
            avg_mutations = sum(s['mutations'] for s in self.stats_history) / len(self.stats_history)
            avg_success_rate = sum(s['success_rate'] for s in self.stats_history) / len(self.stats_history) * 100
            
            print(f"平均基因长度: {avg_gene_length:.1f}")
            print(f"平均变异次数: {avg_mutations:.1f}")
            print(f"平均成功率: {avg_success_rate:.1f}%")
        
        print("\n算法进化总结:")
        print("1. 基因链从初始的50个碱基开始，随着成功决策自然增长")
        print("2. 环境压力动态控制基因表达类型（激进/保守/灵活/探索）")
        print("3. 视觉系统提供实时环境感知，增强决策准确性")
        print("4. X碱基引入创造性探索，突破局部最优")
        print("5. 基因记忆系统记录成功模式，避免重复错误")
        print("=" * 70)

# 运行游戏
if __name__ == "__main__":
    game = GEDAGame()
    game.run()