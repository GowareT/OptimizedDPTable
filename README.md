# OptimizedDPTable

### 一、OptimizedDPTable介绍

&nbsp;&nbsp;`OptimizedDPTable`是用来实现一种优化的DP表算法，同时也可以实现一下功能。
1. 随机生成一维或二维DP表。
2. 传统DP表算法（动态规划）。
3. **优化的DP表算法（贪心算法、动态规划、启发式算法）。**
4. 打印量子电路结构到控制台
5. 对量子电路进行序列化并存储为qasm文件。
6. 计算张量收缩的计算成本。

### 二、参数说明

1. `total contraction cost` 是一个整体性优化结果。 `global total cost `是分层优化结果，可能在某些场景中比全局优化更高效，但其成本高于全局最优值。

### 三、优化的DP表算法思路

&nbsp;&nbsp;最早的DP表张量收缩优化算法是`矩阵链乘法动态规划算法`,但不适用跨层优化和复杂结构。目前在研究的为分层优化方法。因此这个程序主
要思路如下：
1. **分层处理**：将有共享索引的张量分到一层中，降低层内优化复杂度。并且将没有共享索引的张量合并到独立层，等分层结束作为独立张量在后面跨层优化时进行处理。
2. **层内优化**：共享索引可以采用动态规划或者贪心算法进行优化，未共享索引采用动态规划或者启发式算法进行优化
3. **跨层优化**：合并不同层的张量，考虑全局共享索引对计算成本影响。即构建跨层DP表并计算全局最优收缩顺序。


### 四、主函数逻辑（未更新）

1. 调用`CreateRandomDpTable`类中两个方法，`one_dimensional_random_dp_tables()`和`two_dimensional_random_dp_tables()`
生成随机DP表存储在数组中，一维DP表存储在二维数组 ，二维DP表存储在三维数组中。 
2. 调用`SerializedCircuitGenerator`类中`serialize_to_qasm()`对DP表进行序列化并存储为Files目录下的.qasm文件。
3. 调用`SerializedCircuitGenerator.print_circuit()`将量子的门用张量链表示并打印电路结构到控制台。
4. 调用`CoreProgram.print_tensor_chain_structure()`对上面1的函数返回的DP表数组进行传参，即使用传统DP表算法`矩阵链乘法动态规划算法`,
并调用`CoreProgram.calculate_tensor_contraction_cost()`和`CoreProgram.calculate_total_cost()`分别在控制台输出`总收缩成本`和
`显示最优收缩顺序`。
5. 调用`OptimizeDp`类，对上面1的函数返回的DP表数组进行传参，即使用优化后的DP表算法，调用`CoreProgram.calculate_tensor_contraction_cost()`和`CoreProgram.calculate_total_cost()`分别在控制台输出`总收缩成本`和
`显示最优收缩顺序`。


### 五、页面结构

请将主要函数方法写在core.py中。  

optimized-dptable/  
|  
|-- main.py  &nbsp;&nbsp; # 开始文件  
|-- README.en.md &nbsp;&nbsp;&nbsp; # 项目说明文件英文版  
|-- README.md  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # 项目说明文件  
|-- requirements.txt  &nbsp;&nbsp;&nbsp;&nbsp; # 依赖包  
|-- .gitignore   &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;  # Git忽略规则  
|-- optimized_dp_table/  &nbsp;&nbsp; # 主项目文件  
|&nbsp;&nbsp;&nbsp;&nbsp;|-- create_dp.py  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # 创建随机DP表  
|&nbsp;&nbsp;&nbsp;&nbsp;|-- new_dp_algorithm.py  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # 传统DP表算法  
|&nbsp;&nbsp;&nbsp;&nbsp;|-- old_dp_algorithm.py  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # 优化DP表算法  
|&nbsp;&nbsp;&nbsp;&nbsp;|-- serializers.py  &nbsp;&nbsp;&nbsp;&nbsp; # 序列化和保存为qasm文件  
|&nbsp;&nbsp;&nbsp;&nbsp;|-- setting.py  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # 配置  
|-- qasm_files/  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # 导出的qasm目录  

页面及类和方法之间的层级关系如下：
![img.png](img.png)

### 六、更新记录

1. 初次完成

### 其他

更多等待完善  

@Author:goware  
@email：goware@163.com