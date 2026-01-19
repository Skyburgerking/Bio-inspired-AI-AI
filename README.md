# Bio-inspired-AI (with original algorithm)
探索生物启发的人工智能新范式。核心：基因表达决策算法(GEDA)、跨语言语音语义网络、联想记忆系统。目标是实现能持续进化、具备认知手感的智能体。with original algorithm
🧬 Bio-inspired AI: GEDA Visual Snake
🧬 生物启发AI：基因表达决策算法视觉贪吃蛇
English | 中文
A bio-inspired artificial intelligence project that simulates gene expression decision-making in a visual snake game. This project combines computational biology, genetic algorithms, and visual perception to create an adaptive AI agent.

Author: Skyburgerking

🎮 Project Overview
GEDA (Gene Expression Decision Algorithm) is an AI model inspired by biological gene expression and protein synthesis. The system uses a five-base genetic code to control decision-making in a classic Snake game, with real-time visual perception and environmental pressure sensing.

✨ Key Features
Five-Base Genetic System:

A: Aggressive action (seeking food)

G: Stable operation (maintaining safety)

C: Conservative strategy (avoiding risks)

T: Flexible adjustment (adaptive behavior)

X: Exploratory mutation (creative exploration)

Visual Perception System:

Real-time gene expression map visualization

Environmental pressure heatmap

Visual memory and cognitive mapping

Adaptive Decision-Making:

Dynamic gene activation based on environmental pressure

A* pathfinding integrated with genetic decisions

Memory-based learning from past experiences

Evolutionary Mechanics:

5 mutation types: insertion, replacement, deletion, reordering, complement

Gene chain growth through successful decisions

Natural selection simulated through gameplay

🚀 Getting Started
Prerequisites
Python 3.7+

Pygame library

Installation
bash
# Clone the repository
git clone https://github.com/Skyburgerking/Bio-inspired-AI.git

# Navigate to project
cd Bio-inspired-AI

# Install dependencies
pip install pygame numpy
Running the Game
bash
python geda_snake_game.py
🧪 How It Works
Gene Chain Initialization: The AI starts with a 50-base genetic chain containing different behavioral segments.

Environmental Sensing: The system calculates pressure from hunger, space constraints, food proximity, and gene diversity.

Gene Expression: Environmental pressure activates specific gene segments (aggressive, conservative, balanced).

Decision Mapping: The active gene segment is mapped to movement decisions using:

A* pathfinding for efficient food pursuit

Safety evaluation for obstacle avoidance

Exploratory behavior from X-bases

Evolution & Learning:

Successful decisions reinforce current genetic patterns

Environmental stress triggers mutations

Memory databases store successful gene expressions

📊 Performance Metrics
The game tracks multiple AI performance indicators:

Decision success rate

Environmental pressure levels

Gene chain length and composition

Mutation count and types

Expression frequency

🎯 Controls
Space: Pause/Resume game

R: Restart with next generation (evolution)

Game runs autonomously - watch the AI learn!

📈 Evolutionary Results
Over multiple generations, the AI demonstrates:

Increasing average scores through genetic optimization

Adaptive gene composition changes based on environment

Emergence of successful behavioral patterns

Creative exploration breaking local optima

🤝 Contributing
This project is open for contributions! Areas for improvement:

Enhanced genetic encoding schemes

Additional environmental factors

More sophisticated mutation mechanisms

Performance optimization

Extended visualization features


中文
🎮 项目概述
GEDA（基因表达决策算法） 是一个受生物基因表达和蛋白质合成启发的AI模型。该系统使用五进制遗传密码来控制经典贪吃蛇游戏中的决策过程，具备实时视觉感知和环境压力感应能力。

作者： Skyburgerking

✨ 核心特性
五进制遗传系统：

A: 激进行动（积极寻找食物）

G: 稳定操作（保持安全距离）

C: 保守策略（规避风险）

T: 灵活调整（适应性行为）

X: 突变探索（创造性探索）

视觉感知系统：

实时基因表达地图可视化

环境压力热力图

视觉记忆与认知地图

自适应决策：

基于环境压力的动态基因激活

A*寻路算法与遗传决策结合

基于记忆的过往经验学习

进化机制：

5种变异类型：插入、替换、删除、重排、互补

通过成功决策实现基因链自然生长

通过游戏玩法模拟自然选择

🚀 快速开始
环境要求
Python 3.7+

Pygame 库

安装
bash
# 克隆仓库
git clone https://github.com/Skyburgerking/Bio-inspired-AI.git

# 进入项目目录
cd Bio-inspired-AI

# 安装依赖
pip install pygame numpy
运行游戏
bash
python geda_snake_game.py
🧪 工作原理
基因链初始化：AI从包含不同行为片段的50碱基基因链开始。

环境感知：系统计算来自饥饿、空间限制、食物接近度和基因多样性的压力。

基因表达：环境压力激活特定基因片段（激进型、保守型、平衡型）。

决策映射：通过以下方式将活跃基因片段映射为移动决策：

使用A*算法高效寻找食物

安全评估规避障碍

X碱基带来的探索行为

进化与学习：

成功决策强化当前遗传模式

环境压力触发基因突变

记忆数据库存储成功的基因表达

📊 性能指标
游戏追踪多种AI性能指标：

决策成功率

环境压力水平

基因链长度与组成

突变次数与类型

基因表达频率

🎯 控制说明
空格键：暂停/继续游戏

R键：重启下一代（进化）

游戏自主运行 - 观看AI学习过程！

📈 进化结果
经过多代进化，AI展现出：

通过基因优化实现平均分数提升

基于环境的适应性基因组成变化

成功行为模式的涌现

创造性探索突破局部最优

🤝 贡献指南
本项目开放贡献！可改进的领域包括：

增强型遗传编码方案

更多环境因素

更复杂的突变机制

性能优化

扩展可视化功能

