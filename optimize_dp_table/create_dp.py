# -*- coding: utf-8 -*-
"""
@Author: Goware
@Date: 2024/11/23
@Description: 创建DP表（一维和二维）
"""
import random
from optimize_dp_table.setting import QUANTUMGATES


# 随机生成DP表，并返回的为一维或二维 DP 表
class CreateRandomDpTable:
    def __init__(self, num_tables, num_qubits, min_length, max_length):
        """
        初始化 SerializedCircuitGenerator 实例。

        Attributes:
            num_tables (int): 生成的DP表数量。
            num_qubits (int): 每个 DP 表中的量子比特数量（行数）。
            min_length (int): DP表中量子门序列最小长度。
            max_length (int): DP表中量子门序列最大长度。

        Methods:
            one_dimensional_random_dp_tables():生成指定数量随机一维DP表
            two_dimensional_random_dp_tables():生成指定数量随机二维DP表
        """

        # 验证输入字段的正确性
        if not all(
                isinstance(i, int) and i > 0
                for i in [num_tables, num_qubits, min_length, max_length]
        ):
            # 如果不是，则抛出ValueError异常
            raise ValueError("All parameters must be positive integers")
        # 检查min_length是否大于max_length
        if min_length > max_length:
            # 如果是，则抛出ValueError异常
            raise ValueError("Min_Length cannot be greater than Max_Length.")

        self.num_tables = num_tables
        self.num_qubits = num_qubits
        self.min_length = min_length
        self.max_length = max_length
        # 常用的量子电路门
        self.quantum_gates = QUANTUMGATES

    def one_dimensional_random_dp_tables(self):
        """
        生成一维DP表
        """
        # 用来存储一维 DP 表
        dp_tables = []
        for _ in range(self.num_tables):
            # 随机生成每个 DP 表的长度
            length = random.randint(self.min_length, self.max_length)
            # 随机选择量子门操作构建 DP 表
            dp_table = [random.choice(self.quantum_gates) for _ in range(length)]
            dp_tables.append(dp_table)
        # 返回的是二维列表，每个子列表示一个一维DP表
        print("one_dimensional：", dp_tables)
        return dp_tables

    # 生成指定数量的二维DP表
    def two_dimensional_random_dp_tables(self):
        """
        生成二维DP表
        """
        dp_tables = []  # 初始化 DP 表列表，存储多个二维 DP 表
        for _ in range(self.num_tables):
            # 用来存储一个二维 DP 表
            dp_table = []
            for _ in range(self.num_qubits):
                # 随机生成当前量子比特上的量子门序列
                length = random.randint(self.min_length, self.max_length)
                dp_table.append(
                    [random.choice(self.quantum_gates) for _ in range(length)]
                )
            dp_tables.append(dp_table)
        # 返回的是二维列表，每个子列表示一个二维DP表
        print("two_dimensional：", dp_tables)
        return dp_tables
