# -*- coding: utf-8 -*-
"""
@Author: Goware
@Date: 2024/11/20
@Description: 序列化
"""
import os
from qiskit.qasm3 import dump

from qiskit import QuantumCircuit, transpile

from optimize_dp_table.setting import DPTABLES_TYPE


class SerializedProgram:
    """
    对DP表进行序列化,向控制台打印量子电路、张量链表示并导出

    Attributes:
        dp_tables (list): 生成的DP表

    Methods:
        construct_circuit(): 根据dp_tables生成指定数量子电路
        construct_tensor_chain(): 根据dp_tables表生成张量链表示
        _apply_gate(): 根据操作类型向量子电路中添加对应的门
        export_qasm(): 序列化 DP 表生成的量子电路并保存为 QASM 文件。
        print_circuit(): 打印生成的量子电路到控制台；生成张量链并表示
    """

    def __init__(self, dp_tables):
        # 初始化 DP 表类型和 DP 表
        self.dp_types = DPTABLES_TYPE
        self.dp_tables = dp_tables

    def preprocess_dp_table(self):
        """
        将 DP 表统一为张量形状列表，支持一维和二维输入格式。
        :param dp_table: DP 表，可能是一维或二维的张量列表。
        :return: 统一的张量形状列表 [(rows, cols), ...]。
        """
        if not self.dp_tables:
            raise ValueError("DP table is empty or invalid.")
        tensor_shapes = []
        # 判断是否为一维 DP 表：每个元素是一个元组 (rows, cols)
        if all(isinstance(tensor, tuple) and len(tensor) == 2 for tensor in self.dp_tables):
            # 直接使用一维 DP 表的张量形状
            tensor_shapes = self.dp_tables
        # 判断是否为二维 DP 表：每个元素是一个列表（表示张量的行）
        elif all(isinstance(tensor, list) and isinstance(tensor[0], list) for tensor in self.dp_tables):
            for tensor in self.dp_tables:
                rows = len(tensor)  # 行数
                cols = max(len(row) for row in tensor) if tensor else 0  # 最大列数
                tensor_shapes.append((rows, cols))
        else:
            raise ValueError("Unsupported DP table format. DP table must be one-dimensional or two-dimensional.")
        return tensor_shapes

    def construct_circuit(self):
        """
        根据 DP 表生成量子电路，支持一维和二维 DP 表。
        """
        # 获取量子比特数
        num_qubits = len(self.dp_tables)
        # 创建量子电路
        circuit = QuantumCircuit(num_qubits)
        # 判断DP表类型
        if self.dp_types == "one_dimensional":
            # 一维 DP 表处理逻辑
            for qubit_idx, operations in enumerate(self.dp_tables):
                for operation in operations:
                    # 应用门操作
                    self._apply_gate(circuit, qubit_idx, operation, num_qubits)
        elif self.dp_types == "two_dimensional":
            # 二维 DP 表处理逻辑
            for qubit_idx, time_steps in enumerate(self.dp_tables):
                for operations in time_steps:
                    for operation in operations:
                        # 应用门操作
                        self._apply_gate(circuit, qubit_idx, operation, num_qubits)
        else:
            # 抛出错误
            raise ValueError(
                "Unsupported DP table type. Use 'one_dimensional' or 'two_dimensional'."
            )
        # 返回量子电路
        return circuit

    def construct_tensor_chain(self):
        """
        根据 DP 表生成张量链表示。
        """
        tensor_chain = []
        # 如果dp_types为"one_dimensional"，则遍历dp_tables中的每个操作
        if self.dp_types == "one_dimensional":
            for qubit_idx, operations in enumerate(self.dp_tables):
                for operation in operations:
                    tensor_chain.append((operation, (qubit_idx,)))
        # 如果dp_types为"two_dimensional"，则遍历dp_tables中的每个时间步
        elif self.dp_types == "two_dimensional":
            for qubit_idx, time_steps in enumerate(self.dp_tables):
                for operations in time_steps:
                    for operation in operations:
                        tensor_chain.append((operation, (qubit_idx,)))
        # 如果dp_types不是"one_dimensional"或"two_dimensional"，则抛出错误
        else:
            raise ValueError(
                "Unsupported DP table type. Use 'one_dimensional' or 'two_dimensional'."
            )
        return tensor_chain

    def _apply_gate(self, circuit, qubit_idx, operation, num_qubits):
        """
        根据操作类型向量子电路中添加对应的门。
        """
        if operation == "H":
            circuit.h(qubit_idx)
        elif operation == "CNOT":
            target = (qubit_idx + 1) % num_qubits
            circuit.cx(qubit_idx, target)
        elif operation == "RX":
            circuit.rx(3.14, qubit_idx)
        elif operation == "RY":
            circuit.ry(3.14, qubit_idx)
        elif operation == "CZ":
            target = (qubit_idx + 1) % num_qubits
            circuit.cz(qubit_idx, target)
        elif operation == "X":
            circuit.x(qubit_idx)
        elif operation == "Y":
            circuit.y(qubit_idx)
        elif operation == "Z":
            circuit.z(qubit_idx)
        elif operation == "T":
            circuit.t(qubit_idx)
        elif operation == "S":
            circuit.s(qubit_idx)
        elif operation == "CX":
            target = (qubit_idx + 1) % num_qubits
            circuit.cx(qubit_idx, target)
        elif operation == "CY":
            target = (qubit_idx + 1) % num_qubits
            circuit.cy(qubit_idx, target)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def export_qasm(self, filename):
        """
        序列化 DP 表生成的量子电路并保存为 QASM 文件。
        """
        try:
            output_dir = "qasm_files"
            os.makedirs(output_dir, exist_ok=True)
            # 构建量子电路并验证
            circuit = self.construct_circuit()
            # 序列化为 QASM 文件
            filepath = os.path.join(output_dir, filename)  # 拼接文件路径
            with open(filepath, "w") as f:
                # 导入时请使用 QASM的dump 格式化函数
                dump(circuit, f)
            print(f"QASM file generated successfully: {filepath}")
        except Exception as e:
            print(f"Failed to write QASM file: {e}")

    def print_circuit(self):
        """
        打印生成的量子电路到控制台；生成张量链并表示
        """
        # 构建电路
        circuit = self.construct_circuit()
        # 打印电路
        print(circuit.draw())
        # 构建张量链
        tensor_chain = self.construct_tensor_chain()
        # 打印张量链表示
        print("\nTensor Chain Representation:")
        # 遍历张量链
        for tensor in tensor_chain:
            # 获取操作和量子比特
            operation, qubits = tensor
            # 将量子比特转换为字符串
            qubits_str = ", ".join(map(str, qubits))
            # 打印张量
            print(f"Tensor(Operation={operation}, Qubits=({qubits_str}))")

