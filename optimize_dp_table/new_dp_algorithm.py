# -*- coding: utf-8 -*-
"""
@Author: Goware
@Date: 2024/11/20
@Description:优化DP表算法中总收缩成本、总计算成本和显示最优顺序。
"""
from optimize_dp_table.setting import SHARED_INDEX_LAYER_AIG, INDEPENDENT_INDEX_LAYER


# 主要业务调用逻辑
class NewDpAlgorithmProgram:
    def __init__(self, dp_tables):
        """
        初始化 NewDpAlgorithmProgram 实例。
        """
        self.dp_tables = dp_tables

    # 使用优化DP算法输出总收缩成本和显示最优收缩顺序
    def new_dp_algorithm_lg(self):
        """
        对二维 DP 表执行优化算法（分层 + 层内优化 + 跨层优化）。
        """
        # 1. 分层处理
        layers = self.split_into_layers()
        print('分层后的张量集合，layers', layers)
        # 2. 层内优化
        total_computation_cost = 0  # 初始化总计算成本
        layer_results = []
        for idx, layer in enumerate(layers):
            if not layer:  # 跳过空层
                continue
            # 判断层类型
            layer_type = self.determine_layer_type(layer)
            if layer_type == "shared":
                print(SHARED_INDEX_LAYER_AIG, '共享层已分层')
                # 如果是有共享索引层则采用动态规划或贪心算法
                if SHARED_INDEX_LAYER_AIG == 'DP':
                    # 使用动态规划算法优化有共享索引层
                    # 返回的是共享层总收缩成本和最优收缩顺序
                    cost, order = self.optimize_within_layer_dp(layer)
                elif SHARED_INDEX_LAYER_AIG == 'GA':
                    # 使用贪心算法优化有共享索引层
                    # 返回的是共享层总收缩成本和近似收缩顺序
                    cost, order = self.optimize_within_layer_ga(layer)
                else:
                    raise ValueError("Unsupported optimization method: ", SHARED_INDEX_LAYER_AIG)
                print('共享层总收缩成本，total_contraction_cost=',cost)
                print('共享层收缩最优顺序，optimal_order=', order)
                total_computation_cost += cost  # 更新总计算成本
                layer_results.append((cost, order, layer))  # 添加共享层的优化结果
            # 如果该层为独立层
            elif layer_type == "independent":
                # 无共享索引层优化策略
                if INDEPENDENT_INDEX_LAYER == 'DP':
                    # 使用动态规划算法优化无共享索引层即独立层
                    # 返回的是独立层总收缩成本和最优收缩顺序
                    print(INDEPENDENT_INDEX_LAYER,'独立层已分层')
                    independent_cost, independent_computation_cost, independent_order = self.optimize_independent_layer_dp(
                        layer)

                elif INDEPENDENT_INDEX_LAYER == 'HEURISTIC':
                    # 使用启发式算法优化无共享索引层即独立层
                    # 返回的是独立层总收缩成本和最优收缩顺序
                    print('HEURISTIC独立层被执行')
                    independent_cost, independent_computation_cost, independent_order = self.optimize_independent_layer_heuristic(
                        layer)
                else:
                    raise ValueError("Unsupported independent layer optimization method:", INDEPENDENT_INDEX_LAYER)
                total_computation_cost += independent_computation_cost  # 更新总计算成本
                layer_results.append((independent_cost, independent_order, layer))  # 添加独立层的优化结果
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        # 3. 跨层优化
        print("开始进行跨层优化")
        cross_layer_cost, cross_layer_order = self.optimize_across_layers(layer_results)

        # 计算最终总收缩成本和全局最优收缩顺序
        total_contraction_cost = cross_layer_cost

        # 构建全局最优收缩顺序
        optimal_cross_layer_order = cross_layer_order
        for idx, (_, order, _) in enumerate(layer_results):
            optimal_cross_layer_order = optimal_cross_layer_order.replace(f"Layer{idx}", f"({order})")
        # 输出最终结果
        print("|---***---| 优化DP算法总收缩成本,total_contraction_cost=", total_contraction_cost)
        print("|---***---| 优化DP算法总计算成本,total_computation_cost=", total_computation_cost)
        print("|---***---| 优化DP算法全局最优收缩顺序,optimal_cross_layer_order=", optimal_cross_layer_order)

    def split_into_layers(self):
        """
        分层处理
        分层原则：将所有有共享索引放在一层，没有共享索引放在独立层
        """
        layers = []  # 存储分层结果
        current_layer = []  # 当前层
        current_rows = set()  # 当前层的行索引集合
        current_cols = set()  # 当前层的列索引集合
        for dp_table in self.dp_tables:
            if not dp_table or not all(
                    isinstance(row, list) and len(row) > 0 for row in dp_table
            ):
                continue
            # 计算当前张量的索引集合
            dp_rows, dp_cols = self.get_indices(dp_table)
            # print(f"Current dp_table indices: {dp_indices}")
            # 如果当前层为空，或者与当前层的行或列索引有较大交集，则将张量加入当前层
            threshold = 5  # 假设当索引交集元素数量超过 5 时才认为它们有共享索引
            if not current_layer or len(current_rows & dp_rows) > threshold or len(current_cols & dp_cols) > threshold:
                current_layer.append(dp_table)
                current_rows.update(dp_rows)
                current_cols.update(dp_cols)
            else:
                # 当前张量与当前层无较大共享索引，开始新的层
                layers.append(current_layer)
                current_layer = [dp_table]
                current_rows = dp_rows
                current_cols = dp_cols
            # 添加最后一层
        if current_layer:
            layers.append(current_layer)
        return layers

    def get_indices(self, dp_table):
        """
        计算张量的索引集合。
        对于二维张量，索引集合可以是行或列的范围。
        """
        rows = set(range(len(dp_table)))  # 行索引范围
        cols = set(range(max(len(row) for row in dp_table)))  # 列索引范围
        return rows, cols

    def optimize_within_layer_dp(self, layer):
        """
        使用动态规划优化有共享索引的层。
        :param layer: 层内张量集合，每个张量的形状为 (rows, cols)。
        :return: 总收缩成本和最优收缩顺序。
        """
        # tensor的形状
        tensor_shapes = [(len(tensor), max(len(row) for row in tensor)) for tensor in layer]
        n = len(tensor_shapes)
        # 初始化dp矩阵，dp[i][j]表示将第i个tensor到第j个tensor合并的代价
        dp = [[float("inf")] * n for _ in range(n)]
        # 初始化split矩阵，split[i][j]表示将第i个tensor到第j个tensor合并的分割点
        split = [[0] * n for _ in range(n)]
        # 初始化dp矩阵的对角线元素为0，表示将一个tensor合并的代价为0
        for i in range(n):
            dp[i][i] = 0
        # 遍历所有可能的合并长度
        for length in range(2, n + 1):
            # 遍历所有可能的合并起点
            for i in range(n - length + 1):
                # 计算合并终点
                j = i + length - 1
                # 遍历所有可能的分割点
                for k in range(i, j):
                    # 计算合并代价
                    cost = (
                            dp[i][k]
                            + dp[k + 1][j]
                            + tensor_shapes[i][0] * tensor_shapes[k][1] * tensor_shapes[j][1]
                    )
                    # 如果合并代价小于当前最小代价，则更新最小代价和分割点
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = k
        def construct_order(i, j):
            if i == j:
                return f"T{i}"
            k = split[i][j]
            return f"({construct_order(i, k)} x {construct_order(k + 1, j)})"

        total_cost = dp[0][n - 1]
        optimal_order = construct_order(0, n - 1)
        return total_cost, optimal_order

    def optimize_within_layer_ga(self, layer):
        """
        使用贪心算法优化有共享索引的层。
        :param layer: 层内张量集合，每个张量的形状为 (rows, cols)。
        :return: 总收缩成本和近似收缩顺序。
        """
        tensor_shapes = [(len(tensor), max(len(row) for row in tensor)) for tensor in layer]
        total_cost = 0
        order = []
        while len(tensor_shapes) > 1:
            min_cost = float("inf")
            min_idx = 0

            # 找到当前收缩成本最低的两个张量
            for i in range(len(tensor_shapes) - 1):
                cost = (
                        tensor_shapes[i][0]
                        * tensor_shapes[i][1]
                        * tensor_shapes[i + 1][1]
                )
                if cost < min_cost:
                    min_cost = cost
                    min_idx = i

            # 收缩这两个张量
            total_cost += min_cost
            order.append((min_idx, min_idx + 1))
            new_tensor = (tensor_shapes[min_idx][0], tensor_shapes[min_idx + 1][1])
            tensor_shapes = (
                    tensor_shapes[:min_idx]
                    + [new_tensor]
                    + tensor_shapes[min_idx + 2:]
            )

        # 构造近似收缩顺序
        final_order = " x ".join(f"T{i}" for i, _ in order)
        return total_cost, f"({final_order})"


    def determine_layer_type(self, layer):
        """
        判断层的类型：共享层或独立层。
        :param layer: 单个层，包含多个张量。
        :return: "shared" 表示共享层，"independent" 表示独立层。
        """
        if len(layer) == 1:
            return "independent"  # 只有一个张量，直接为独立层
        # 获取所有张量的索引集合
        indices_list = [self.get_indices(dp_table) for dp_table in layer]
        # 判断是否存在共享索引（行或列共享都视为共享索引）
        for i in range(len(indices_list)):
            for j in range(i + 1, len(indices_list)):
                rows_i, cols_i = indices_list[i]
                rows_j, cols_j = indices_list[j]
                # 判断行或列是否有共享
                if not rows_i.isdisjoint(rows_j) or not cols_i.isdisjoint(cols_j):
                    return "shared"
        return "independent"

    def optimize_independent_layer_dp(self, layer):
        """
        使用动态规划优化无共享索引层。
        :param layer: 无共享索引层张量集合，每个张量为二维数组。
        :return: 总收缩成本、总计算成本、最优收缩顺序。
        """
        total_contraction_cost = 0  # 总收缩成本
        total_computation_cost = 0  # 总计算成本
        optimal_orders = []  # 存储每个张量的最优操作顺序

        for tensor_idx, tensor in enumerate(layer):
            n = len(tensor)
            dp = [[float("inf")] * n for _ in range(n)]
            split = [[0] * n for _ in range(n)]

            for i in range(n):
                dp[i][i] = 0

            for length in range(2, n + 1):  # 长度从2开始
                for i in range(n - length + 1):
                    j = i + length - 1
                    for k in range(i, j):
                        cost = (
                                dp[i][k]
                                + dp[k + 1][j]
                                + len(tensor[i]) * len(tensor[k]) * len(tensor[j])
                        )
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            split[i][j] = k

            # 构造当前张量的最优操作顺序
            def construct_order(i, j):
                if i == j:
                    return f"T{tensor_idx}-{i}"
                k = split[i][j]
                return f"({construct_order(i, k)} x {construct_order(k + 1, j)})"

            contraction_cost = dp[0][n - 1]  # 当前张量的总收缩成本
            optimal_order = construct_order(0, n - 1)

            # 累加总收缩成本和计算成本
            total_contraction_cost += contraction_cost
            total_computation_cost += contraction_cost
            optimal_orders.append(optimal_order)

        # 将所有张量的操作顺序组合为整体的最优操作顺序
        global_optimal_order = " ; ".join(optimal_orders)

        # 打印结果
        print('Dynamic Programming (Independent Layer):')
        print(f"独立层总收缩成本,total_contraction_cost= {total_contraction_cost}")
        print(f"独立层总计算成本,total_cost= {total_computation_cost}")
        print(f"独立层收缩最优顺序,optimal_order= {global_optimal_order}")

        return total_contraction_cost, total_computation_cost, global_optimal_order

    def optimize_independent_layer_heuristic(self, layers):
        """
        使用启发式算法优化无共享索引层。
        :param layers: 无共享索引层张量集合，每个张量为二维数组。
        :return: 总收缩成本、总计算成本、最优收缩顺序。
        """
        total_contraction_cost = 0  # 总收缩成本
        total_computation_cost = 0  # 总计算成本
        optimal_orders = []  # 存储每个张量的收缩顺序

        # 启发式规则：按行数 × 列数排序
        sorted_tensors = sorted(
            enumerate(layers), key=lambda item: len(item[1]) * max(len(row) for row in item[1])
        )

        for tensor_idx, tensor in sorted_tensors:
            rows = len(tensor)
            cols = max(len(row) for row in tensor)
            contraction_cost = rows * cols

            # 累加成本
            total_contraction_cost += contraction_cost
            total_computation_cost += contraction_cost
            optimal_orders.append(f"T{tensor_idx}")

        # 构造全局最优顺序
        global_optimal_order = " x ".join(optimal_orders)

        # 打印结果
        print("Heuristic Optimization (Independent Layer):")
        print(f"独立层总收缩成本,total_contraction_cost= {total_contraction_cost}")
        print(f"独立层总计算成本,total_cost= {total_computation_cost}")
        print(f"独立层收缩最优顺序,optimal_order= {global_optimal_order}")

        return total_contraction_cost, total_computation_cost, global_optimal_order

    def optimize_across_layers(self, layer_results):
        """
        对分层后的张量链进行跨层优化。
        :param layer_results: 每层的优化结果，包括 (cost, order, layer)
        :return: 跨层的总收缩成本和最优收缩顺序。
        """
        if len(layer_results) < 2:
            # 如果只有一层或者没有层，不需要跨层优化
            return layer_results[0][0], layer_results[0][1] if layer_results else (0, "No Layers")

        # 提取各层的张量形状用于跨层收缩
        tensor_shapes = [(len(layer), max(len(row) for tensor in layer for row in tensor)) for _, _, layer in layer_results]
        n = len(tensor_shapes)
        dp = [[float("inf")] * n for _ in range(n)]
        split = [[0] * n for _ in range(n)]
        # 初始化单张量成本为 0
        for i in range(n):
            dp[i][i] = 0
        # 动态规划计算跨层的收缩成本
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
        # 构造跨层的最优收缩顺序
        def construct_cross_layer_order(i, j):
            # 如果i等于j，则返回"Layer{i}"
            if i == j:
                return f"Layer{i}"
            k = split[i][j]
            return f"({construct_cross_layer_order(i, k)} x {construct_cross_layer_order(k + 1, j)})"
        # 计算总跨层层成本
        total_cross_layer_cost = dp[0][n - 1]
        # 构造最优跨层层顺序
        optimal_cross_layer_order = construct_cross_layer_order(0, n - 1)
        # 返回总跨层层成本和最优交叉层顺序
        return total_cross_layer_cost, optimal_cross_layer_order