# -*- coding: utf-8 -*-
"""
@Author: Goware
@Date: 2024/11/20
@Description: 相关配置，全局常量
"""

# 常用的量子电路门
QUANTUMGATES = quantum_gates = [
    "H",  # Hadamard gate
    "CNOT",  # Controlled-NOT gate
    "RX",  # Rotation around X-axis
    "RY",  # Rotation around Y-axis
    "CZ",  # Controlled-Z gate
    "X",  # Pauli-X gate
    "Y",  # Pauli-Y gate
    "Z",  # Pauli-Z gate
    "T",  # T gate
    "S",  # S gate
    "CX",  # Controlled-X gate
    "CY",  # Controlled-Y gate
]

# 随机生成的DP表类型，one_dimensional为一维，two_dimensional为二维，本课题提出的优化算法为二维
DPTABLES_TYPE = 'two_dimensional'
# 随机生成的DP表参数
NUM_TABLES = 20  # 生成 30 个 DP 表
NUM_QUBITS = 1  # 每个表有 1 个量子比特，默认为 1
MIN_LENGTH = 1  # 每个序列最短长度
MAX_LENGTH = 30  # 每个序列最长长度
SHARED_INDEX_LAYER_AIG = 'DP'  # 共享索引层算法,DP 为动态规划算法，GA 为贪心算法
INDEPENDENT_INDEX_LAYER = 'DP'  # 无共享索引层,DP 为动态规划算法，HEURISTIC 为启发式算法
