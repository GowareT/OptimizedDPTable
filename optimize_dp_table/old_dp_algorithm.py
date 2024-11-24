# -*- coding: utf-8 -*-
"""
@Author: Goware
@Date: 2024/11/20
@Description:传统DP表算法中总收缩成本、总计算成本和显示最优顺序。支持一维和二维DP表
"""

from optimize_dp_table.serializers import SerializedProgram


# 主要业务调用逻辑
class OldDpAlgorithmProgram:
    def __init__(self, dp_tables):
        """
        初始化 CoreProgram 实例。
        Attributes:
            dp_tables (int): 生成的DP表。
        Methods:
            old_dp_tensor_contraction_cost():传统DP表算法中输出张量的总收缩成本
            old_dp_total_cost():传统DP表算法中输出张量的总计算成本
            old_dp_optimal_contraction_order():传统DP表算法中输出传统DP表最优收缩顺序
        """
        self.dp_tables = dp_tables
        # 实例化SerializedProgram
        self.dp_serializers = SerializedProgram(self.dp_tables)

    def old_dp_tensor_contraction_cost(self):
        """
        计算每一步张量收缩的成本，支持一维和二维 DP 表。
        计算总张量收缩成本并输出
        """
        # 调用将 DP 表统一为张量形状列表方法，获取张量形状
        tensor_shapes = self.dp_serializers.preprocess_dp_table()
        # 获取张量数量
        n = len(tensor_shapes)
        # 初始化 DP 表
        dp = [[float("inf")] * n for _ in range(n)]
        # 初始化 contraction_costs 列表
        contraction_costs = []
        # 初始化单张量成本为 0
        for i in range(n):
            dp[i][i] = 0
        # 动态规划填充 DP 表
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    # 计算成本
                    cost = (
                            dp[i][k]
                            + dp[k + 1][j]
                            + tensor_shapes[i][0] * tensor_shapes[k][1] * tensor_shapes[j][1]
                    )
                    # 更新 DP 表和 contraction_costs 列表
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        # 在找到更优的成本时，记录该成本
                        if len(contraction_costs) > i:
                            contraction_costs[i] = tensor_shapes[i][0] * tensor_shapes[k][1] * tensor_shapes[j][1]
                        else:
                            contraction_costs.append(tensor_shapes[i][0] * tensor_shapes[k][1] * tensor_shapes[j][1])

        # 返回 contraction_costs 列表
        # 计算总收缩成本
        total_contraction_cost = dp[0][n - 1]  # 使用最终 dp 表的值来获得最优的总收缩成本
        print('|---***---| 传统DP表算法张量总收缩成本,total_contraction_cost=', total_contraction_cost)
        return total_contraction_cost

    def old_dp_total_cost(self):
        """
        计算所有张量收缩的总计算成本，支持一维和二维 DP 表。
        """
        # 获取张量形状
        tensor_shapes = self.dp_serializers.preprocess_dp_table()
        # 获取张量形状的长度
        n = len(tensor_shapes)
        # 初始化dp矩阵，所有元素初始化为无穷大
        dp = [[float("inf")] * n for _ in range(n)]
        # 初始化单张量成本为 0
        for i in range(n):
            dp[i][i] = 0
        # 动态规划填充 DP 表
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    cost = (
                            dp[i][k]
                            + dp[k + 1][j]
                            + tensor_shapes[i][0] * tensor_shapes[k][1] * tensor_shapes[j][1]
                    )
                    if cost < dp[i][j]:
                        dp[i][j] = cost
        # 获取总计算成本传统DP表算法张量总收缩成本
        total_cost = dp[0][n - 1]
        # 打印总计算成本
        print('|---***---| 传统DP表算法张量收缩总计算成本，total_cost=', total_cost)
        return total_cost

    def old_dp_optimal_contraction_order(self):
        """
        计算最优张量收缩顺序，支持一维和二维 DP 表。
        """
        tensor_shapes = self.dp_serializers.preprocess_dp_table()
        n = len(tensor_shapes)
        dp = [[float("inf")] * n for _ in range(n)]
        split = [[0] * n for _ in range(n)]
        # 初始化单张量成本为 0
        for i in range(n):
            dp[i][i] = 0
        # 动态规划填充 DP 表
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    cost = (
                            dp[i][k]
                            + dp[k + 1][j]
                            + tensor_shapes[i][0] * tensor_shapes[k][1] * tensor_shapes[j][1]
                    )
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = k

        # 递归构造最优收缩顺序
        def construct_optimal_order(i, j):
            # 如果i等于j，则返回Ti
            if i == j:
                return f"T{i}"
            # 获取split数组中i和j之间的分割点k
            k = split[i][j]
            # 递归调用construct_optimal_order函数，获取i到k之间的最优排列
            left_order = construct_optimal_order(i, k)
            # 递归调用construct_optimal_order函数，获取k+1到j之间的最优排列
            right_order = construct_optimal_order(k + 1, j)
            # 返回左排列和右排列的组合
            return f"({left_order} x {right_order})"
        # 计算最优收缩顺序
        optimal_order = construct_optimal_order(0, n - 1)
        # 打印最优收缩顺序
        print('|---***---| 传统DP表算法张量收缩最优顺序，optimal_order=', optimal_order)
        return optimal_order
