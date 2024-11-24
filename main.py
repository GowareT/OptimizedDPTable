# -*- coding: utf-8 -*-
"""
@Author: Goware
@Date: 2024/11/20
@Description:启动函数
"""
from optimize_dp_table.create_dp import CreateRandomDpTable
from optimize_dp_table.new_dp_algorithm import NewDpAlgorithmProgram
from optimize_dp_table.old_dp_algorithm import OldDpAlgorithmProgram
from optimize_dp_table.serializers import SerializedProgram
from optimize_dp_table.setting import (
    NUM_TABLES,
    NUM_QUBITS,
    MIN_LENGTH,
    MAX_LENGTH,
    DPTABLES_TYPE,
)

if __name__ == "__main__":
    # 初始化实例
    dp_table_generator = CreateRandomDpTable(
        NUM_TABLES, NUM_QUBITS, MIN_LENGTH, MAX_LENGTH
    )
    # 判断生成二维DP表种类
    if DPTABLES_TYPE == "two_dimensional":
        # 调用生成二维DP表方法
        dp_tables = dp_table_generator.two_dimensional_random_dp_tables()

        # 初始化serializers中SerializedCircuitGenerator实例
        generator = SerializedProgram(dp_tables)
        # 生成量子电路打印到控制台，保存量子电路为 QASM 文件,qasm_filename为文件名
        generator.print_circuit()
        qasm_filename = "initial_quantum_circuit.qasm"
        generator.export_qasm(qasm_filename)

        # 调用传统 DP 表算法
        old_dp_algorithm = OldDpAlgorithmProgram(dp_tables)
        # 输出传统DP表算法张量的总收缩成本
        old_dp_algorithm.old_dp_tensor_contraction_cost()
        # 输出传统DP表算法中输出张量的总计算成本
        old_dp_algorithm.old_dp_total_cost()
        # 输出传统DP表最优收缩顺序
        old_dp_algorithm. old_dp_optimal_contraction_order()

        # 调用优化后 DP 表算法
        new_dp_algorithm = NewDpAlgorithmProgram(dp_tables)
        new_dp_algorithm.new_dp_algorithm_lg()
    elif DPTABLES_TYPE == "one_dimensional":
        # 调用生成一维DP表方法
        dp_tables = dp_table_generator.two_dimensional_random_dp_tables()
        # 暂时不做一维DP方法处理
        pass
    else:
        raise ValueError("Unsupported DP table format")
